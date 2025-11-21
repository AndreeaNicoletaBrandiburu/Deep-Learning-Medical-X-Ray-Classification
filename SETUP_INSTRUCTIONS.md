# Setup Instructions - Pas cu Pas

## 🎯 Ce Trebuie Să Faci Azi

### Pasul 1: Setup Repository (5 minute)

```bash
# Creează repository pe GitHub
# Clone local sau creează direct local
cd d:\Deep_learning_proj

# Structure este deja creată, verifică:
ls src/
```

### Pasul 2: Install Dependencies (2 minute)

```bash
# Creează virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Instalează dependențe
pip install -r requirements.txt
```

### Pasul 3: Download Dataset (10 minute)

1. Mergi la: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Descarcă dataset-ul (va fi un zip)
3. Dezarhivează în proiect:
   ```
   data/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

**Important**: Dataset-ul este mare (~1GB). Asigură-te că ai spațiu și că l-ai pus în `data/` exact cu structura de mai sus.

### Pasul 4: Test Rapid (5 minute)

```bash
# Rulează training pentru 3-5 epoci (test)
python run_resnet.py
```

Acest script:
- Va verifica că dataset-ul este ok
- Va antrena modelul ResNet-18 pentru 5 epoci
- Va salva modelul în `best_resnet18_model.pth`
- Va afișa metrici pe test set

### Pasul 5: Verifică Rezultatele

După training, ar trebui să vezi:
- Metrici de training și validation (loss, accuracy, AUC)
- Model salvat
- Evaluare pe test set cu metrici complete

## ✅ Checklist pentru GitHub

După ce rulezi primul test:

- [ ] Repository creat pe GitHub
- [ ] Toate fișierele pushed (exceptând `data/` și `*.pth`)
- [ ] README.md este actualizat
- [ ] requirements.txt este complet
- [ ] .gitignore exclude dataset-ul și modelele

## 🚀 Next Steps (După Primul Run)

### 1. Training Complet (30-50 epoci)

Editează `run_resnet.py` și schimbă:
```python
epochs = 50  # în loc de 5
```

### 2. Test Vision Transformer

```bash
python run_vit.py
```

### 3. Generate Grad-CAM

După ce ai un model antrenat:
```bash
python example_gradcam.py
```

### 4. Test Compression

```bash
python example_compression.py
```

## 📝 Ce Să Documentezi în README

După training, actualizează README.md cu:
- Metrici reale (ex: Accuracy: 94.5%, ROC-AUC: 0.97)
- Screenshot-uri cu Grad-CAM visualizations
- Comparație ResNet vs ViT (dacă ai antrenat ambele)
- Rezultate compression (reducere size, impact pe accuracy)

## 🎯 Pentru CV

După ce ai rezultate:

**Titlu proiect**: Multimodal Veterinary-Inspired Radiograph Classifier

**Descriere scurtă**:
- Deep learning pipeline for chest X-ray classification (PyTorch)
- Implemented ResNet-18 and Vision Transformer models
- Added Grad-CAM explainability and model compression
- Achieved 94.5% accuracy, 0.97 ROC-AUC

**Link GitHub**: [link-ul tău]

## ⚠️ Probleme Comune

**Eroare: "No module named 'src'"**
- Rulează din directorul rădăcină al proiectului
- Verifică că ai activat virtual environment-ul

**Eroare: "Dataset not found"**
- Verifică că `data/` conține `train/`, `val/`, `test/`
- Verifică că fiecare are subdirectoare `NORMAL/` și `PNEUMONIA/`

**Eroare CUDA: "Out of memory"**
- Redu batch_size în `run_resnet.py` (ex: de la 32 la 16)

**Training prea lent**
- Verifică că folosești GPU: `torch.cuda.is_available()` ar trebui să fie `True`
- Reduce num_workers în `get_dataloaders` dacă ai probleme

## 💡 Tips

1. **Start small**: Testează cu 3-5 epoci mai întâi
2. **Monitor GPU**: `nvidia-smi` să vezi utilizarea GPU
3. **Save frequently**: Modelul se salvează automat cu best weights
4. **Logs**: Toate metricile sunt printate în consolă

## 🎉 Succes!

După ce rulezi primul test cu succes, proiectul este deja prezentabil pe GitHub. Restul (ViT, Grad-CAM, compression) sunt doar straturi bonus care te ridică și mai mult!

