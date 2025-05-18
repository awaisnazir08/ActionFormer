# Temporal Action Detection with OpenTAD + ActionFormer

This project provides a complete pipeline for **Temporal Action Detection (TAD)** using the [OpenTAD](https://github.com/OpenAction/Opentad) framework and [ActionFormer](https://github.com/OpenAction/ActionFormer). It includes feature extraction, training, inference, and evaluation on datasets formatted like ActivityNet.

---

## 📦 Installation

### 1. Install OpenMMLab Dependencies

```bash
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose 
```

### 2. Install MMACTION2

```bash
git clone https://github.com/open-mmlab/mmaction2
cd mmaction2
pip install -v -e .
cd ..
```

### 3. Install OpenTAD

```bash
git clone https://github.com/OpenAction/OpenTAD
cd OpenTAD
pip install -r requirements.txt
cd ..
```

## 📁 Dataset Preparation

### 1. Create the dataset directory:

```bash
mkdir -p datasets/videos
```

### 2. Inside the datasets/ folder, place the following files:

## 📄 `annotations.json`

### ActivityNet-style annotations.  
**Example format:**

```json
{
  "database": {
    "S1_Cheese_C1": {
      "duration": 52.67,
      "frame": 790,
      "annotations": [
        { "segment": [0.87, 4.73], "label": "take" },
        { "segment": [6.67, 11.47], "label": "take" },
        { "segment": [11.47, 18.0], "label": "open" }
      ],
      "subset": "training"
    }
  }
}
```

## 📄 category.idx
### A list of action class names, with one action per line.

## 📄 missing_files.txt
### Optional. A list of video file names that are missing.

## 📁 `videos/`

A directory containing all video files referenced in `annotations.json`.

---

## 🎞️ Feature Extraction

Run the `I3D_Feature_Extraction.ipynb` notebook located inside the `datasets/` folder.  
This will generate a `features/` directory containing `.npz` feature files for each video.

---

## 📥 Pretrained Checkpoint

Download the pretrained ActionFormer checkpoint:  
👉 [ActionFormer Checkpoint (Google Drive)](https://drive.google.com/file/d/1zTWLAerk5lZscOE-RZN9vuZ47MJCDno8/view)

---

## 🚀 Training

From the `OpenTAD/` directory, run:

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  tools/custom_train.py \
  configs/actionformer/thumos_i3d.py \
  --resume path/to/checkpoint.pth
```

## Testing
```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  tools/test.py \
  configs/actionformer/thumos_i3d.py \
  --checkpoint exps/gtea/actionformer_i3d_finetune/gpu1_id0/checkpoint/best.pth
```

**Outputs:**

- `logs.json`
- `results.json`

These will be saved in the `work_dir` specified in `thumos_i3d.py`.

---

## 📊 Evaluation

To compute metrics such as accuracy, F1 score, and precision/recall at IoU thresholds of 10%, 25%, and 50%, run:

```bash
python evaluate.py \
  --results path/to/results.json \
  --labels path/to/original_labels.json \
  --score_threshold 0.45
```

## ✅ Project Structure
.
├── mmaction2/
├── OpenTAD/
├── datasets/
│ ├── videos/
│ ├── annotations.json
│ ├── category.idx
│ ├── missing_files.txt
│ └── I3D_Feature_Extraction.ipynb

---

## 📌 Notes

- Ensure that all filenames in `annotations.json` match the actual filenames in `datasets/videos/`.
- Modify `configs/actionformer/thumos_i3d.py` as necessary to reflect the correct dataset and feature paths.