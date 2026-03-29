import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- CONFIGURAÇÕES GERAIS ---
# Como o script está na pasta 'model', voltamos um nível para acessar 'data'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'PetImages')
IMG_SIZE = (32, 32) # Tamanho clássico da LeNet-5
BATCH_SIZE = 32
EPOCHS = 20

def clean_corrupted_images(folder_path):
    """
    Percorre as pastas e deleta imagens que não podem ser abertas ou estão corrompidas.
    """
    print(f"Verificando imagens corrompidas em: {folder_path}...")
    num_skipped = 0
    for class_name in ['Cat', 'Dog']:
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            filepath = os.path.join(class_dir, filename)
            try:
                # Tenta abrir a imagem e verificar o cabeçalho
                img = Image.open(filepath)
                img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Imagem corrompida deletada: {filepath}")
                os.remove(filepath)
                num_skipped += 1
                
    print(f"Limpeza concluída. {num_skipped} imagens corrompidas removidas.")

def build_lenet5():
    """
    Cria a arquitetura LeNet-5 adaptada para imagens coloridas (3 canais) e 
    usando ReLU/MaxPool (práticas mais modernas que Tanh/AvgPool originais).
    """
    model = models.Sequential([
        # Camada 1: Convolução e Pooling
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Camada 2: Convolução e Pooling
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Achatamento (Flatten) para conectar com a rede neural densa
        layers.Flatten(),
        
        # Camadas Densas (Fully Connected)
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        
        # Camada de Saída (1 neurônio com Sigmoid para classificação binária: Cachorro vs Gato)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 1. Limpar os dados
    clean_corrupted_images(DATA_DIR)

    # 2. Carregar os dados dividindo em Treino (80%) e Validação (20%)
    print("Carregando dataset de treinamento...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    print("Carregando dataset de validação...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # 3. Normalizar os dados (converter pixels de 0-255 para 0-1)
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Otimização de performance (Cache e Prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. Criar o modelo
    model = build_lenet5()
    model.summary()

    # 5. Configurar Callbacks (Boas práticas)
    # EarlyStopping para o treino se a precisão de validação parar de melhorar
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, # Para se não melhorar após 3 épocas
        restore_best_weights=True
    )

    # 6. Treinar o modelo
    print("Iniciando o treinamento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    # 7. Salvar o modelo final
    model_save_path = os.path.join(BASE_DIR, 'model', 'lenet5_cats_dogs.h5')
    model.save(model_save_path)
    print(f"Modelo salvo com sucesso em: {model_save_path}")

if __name__ == "__main__":
    main()