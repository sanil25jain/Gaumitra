import os
from datetime import datetime
from functools import wraps

# --- Core Flask & Web App Imports ---
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import click

# --- Database Imports ---
from pymongo import MongoClient
from bson import ObjectId

# --- ML Model Imports ---
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import json

# =============================================================================
# 1. APPLICATION SETUP & CONFIGURATION
# =============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this' # IMPORTANT: Change for production
app.config['UPLOAD_FOLDER'] = './static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =============================================================================
# 2. DATABASE CONNECTION & SETUP
# =============================================================================
client = MongoClient('mongodb://localhost:27017/')
db = client['bpa_system']

@app.cli.command("init-db")
def init_db_command():
    """CLI command to initialize the database with sample data."""
    print("Initializing the database...")
    for collection in db.list_collection_names():
        db[collection].drop()

    db.create_collection('users')
    db.create_collection('uploads')
    print("✅ Database collections created.")

@app.cli.command("import-images")
@click.command(help="Imports breed images from the data directory into the database.")
def import_images_command():
    """
    Scans the data directory, populates the 'breeds' collection,
    and then populates the 'images' collection with file paths,
    linking them to the correct breed.
    """
    print("Starting image import process...")
    
    # Define the root directory where breed folders are located
    image_root_dir = os.path.join('data', 'Indian_bovine_breeds')
    
    if not os.path.exists(image_root_dir):
        print(f"Error: Directory not found at '{image_root_dir}'. Please check the path.")
        return

    # Clear existing collections for a clean import
    db.breeds.drop()
    db.images.drop()
    print("-> Cleared existing 'breeds' and 'images' collections.")

    total_breeds = 0
    total_images = 0

    # Iterate through each item in the root directory
    for breed_name in os.listdir(image_root_dir):
        breed_path = os.path.join(image_root_dir, breed_name)
        
        # Check if it's a directory (i.e., a breed folder)
        if os.path.isdir(breed_path):
            total_breeds += 1
            images_in_breed = 0
            
            # 1. Create a document for this breed in the 'breeds' collection
            breed_doc = {
                "breed_name": breed_name,
                "description": "No description available." # You can update this later
            }
            result = db.breeds.insert_one(breed_doc)
            breed_id = result.inserted_id # Get the unique ID for this new breed

            # 2. Now, iterate through all files in this breed's folder
            for filename in os.listdir(breed_path):
                # Check if the file is a common image type
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(breed_path, filename)
                    
                    # 3. Create a document for this image in the 'images' collection
                    image_doc = {
                        "breed_id": breed_id,       # Link to the breeds collection
                        "breed_name": breed_name,   # Store name for convenience
                        "image_path": image_path.replace("\\", "/") # Standardize path separators
                    }
                    db.images.insert_one(image_doc)
                    images_in_breed += 1
                    total_images += 1
            
            print(f"-> Processed breed: '{breed_name}' ({images_in_breed} images)")

    print("\n------------------------------------")
    print("✅ Image import complete!")
    print(f"Total breeds imported: {total_breeds}")
    print(f"Total images imported: {total_images}")
    print("------------------------------------")

@app.cli.command("update-breeds")
@click.command(help="Updates the breeds collection with details from breeds_info.json.")
def update_breeds_command():
    """
    Loads data from breeds_info.json and updates the corresponding
    documents in the 'breeds' collection.
    """
    print("Starting breed information update process...")

    # --- Load Breed Info from JSON file ---
    try:
        with open('breeds_info.json') as f:
            breeds_info = json.load(f)
        print(f"-> Successfully loaded {len(breeds_info)} breeds from breeds_info.json.")
    except FileNotFoundError:
        print("Error: breeds_info.json not found. Please ensure it is in the project directory.")
        return
    except json.JSONDecodeError:
        print("Error: Could not parse breeds_info.json. Please check its format.")
        return

    updated_count = 0
    not_found_count = 0

    # Iterate through each breed in the JSON file
    for breed_name, details in breeds_info.items():
        # Find the corresponding breed in the database
        result = db.breeds.update_one(
            {'breed_name': breed_name},  # The filter to find the correct document
            {'$set': {                 # The $set operator updates fields or adds them if they don't exist
                'description': details.get('description', 'N/A'),
                'purpose': details.get('purpose', 'N/A'),
                'milk_yield': details.get('milk_yield', 'N/A'),
                'region': details.get('region', 'N/A'),
                'physical_traits': details.get('physical_traits', 'N/A'),
                'management_tips': details.get('management_tips', 'N/A')
            }}
        )

        if result.matched_count > 0:
            updated_count += 1
        else:
            not_found_count += 1
            print(f"   - Warning: Breed '{breed_name}' from JSON not found in the database. It was skipped.")

    print("\n------------------------------------")
    print("✅ Breed update process complete!")
    print(f"Successfully updated: {updated_count} breeds.")
    if not_found_count > 0:
        print(f"Breeds skipped (not in DB): {not_found_count}.")
    print("------------------------------------")

# =============================================================================
# 3. MACHINE LEARNING MODEL SETUP
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load Breed Info and Classes ---
try:
    with open('breeds_info.json') as f:
        breeds_info = json.load(f)
    classes = list(breeds_info.keys())
    NUM_CLASSES = len(classes)
except FileNotFoundError:
    print("ERROR: breeds_info.json not found. Please place it in the project directory.")
    breeds_info = {}
    classes = []
    NUM_CLASSES = 0

# --- Load the Model ---
MODEL_PATH = 'best_model_final.pth'

def load_model():
    if not os.path.exists(MODEL_PATH) or NUM_CLASSES == 0:
        print("ERROR: Model file not found or classes not loaded. Model not loaded.")
        return None
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")
    return model

model = load_model()

# --- Image Preprocessing Function ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# =============================================================================
# 4. USER AUTHENTICATION & SESSIONS
# =============================================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = db.users.find_one({'username': username})

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            flash('You were successfully logged in!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if db.users.find_one({'username': username}):
            flash('Username already exists.', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            db.users.insert_one({'username': username, 'password_hash': hashed_password})
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# =============================================================================
# 5. CORE APPLICATION ROUTES
# =============================================================================
@app.route("/")
@login_required
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@login_required
def predict():
    """Handles image upload, prediction, and database logging."""
    if 'image' not in request.files or not request.files['image'].filename:
        flash('No image selected!', 'danger')
        return redirect(url_for('index'))

    if not model:
        flash('Model is not loaded. Cannot perform prediction.', 'danger')
        return redirect(url_for('index'))

    img_file = request.files['image']
    filename = secure_filename(img_file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_path)

    try:
        # Load and preprocess image for prediction
        image = Image.open(img_path).convert('RGB')
        img_tensor = preprocess_image(image)

        # Get prediction from the model
        with torch.no_grad():
            outputs = model(img_tensor)
            predicted_idx = torch.argmax(outputs, 1).item()
            predicted_class = classes[predicted_idx]

        # Log the upload and prediction to the database
        upload_record = {
            "user_id": ObjectId(session['user_id']),
            "image_path": img_path,
            "predicted_breed": predicted_class,
            "uploaded_at": datetime.utcnow()
        }
        result = db.uploads.insert_one(upload_record)
        
        # Redirect to a page showing the result
        return redirect(url_for('show_result', upload_id=str(result.inserted_id)))

    except Exception as e:
        flash(f'An error occurred during prediction: {e}', 'danger')
        return redirect(url_for('index'))


@app.route("/result/<upload_id>")
@login_required
def show_result(upload_id):
    """Displays the result of a specific upload."""
    try:
        upload = db.uploads.find_one({'_id': ObjectId(upload_id)})
        if not upload or str(upload['user_id']) != session['user_id']:
            flash('Result not found or you do not have permission to view it.', 'danger')
            return redirect(url_for('index'))
        
        prediction = upload['predicted_breed']
        prediction_info = breeds_info.get(prediction, {"description": "No information available for this breed."})
        image_path = upload['image_path']

        return render_template('result.html',
                               prediction=prediction,
                               prediction_info=prediction_info,
                               image_path=image_path)
    except Exception:
        flash('Invalid result ID.', 'danger')
        return redirect(url_for('index'))

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    app.run(debug=True)
