import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import our custom functions from the .py files
from data_setup import create_data_generators
from model_builder import build_transfer_model  # <-- CHANGED

# --- 1. Set Parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = 'brain_tumor_dataset'
COLOR_MODE_P2 = 'rgb'  # <-- CHANGED
INPUT_SHAPE_P2 = (224, 224, 3) # <-- CHANGED
LEARNING_RATE = 0.0001 # Same as our best custom model
EPOCHS = 30

# --- 2. Load Data ---
print("Loading Path 2 data...")
train_gen, val_gen = create_data_generators(
    dataset_path=DATASET_PATH,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode=COLOR_MODE_P2
)

# --- 3. Build Model ---
print("Building Path 2 model...")
# We call build_transfer_model() with the 3-channel shape
model_p2 = build_transfer_model(input_shape=INPUT_SHAPE_P2) # <-- CHANGED
model_p2.summary() # Print a summary of the model

# --- 4. Compile Model ---
print("Compiling model...")
model_p2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 5. Define Callbacks ---
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# --- 6. Train Model ---
print("Starting training...")
history_p2 = model_p2.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)


print("Training complete for Path 2.")
