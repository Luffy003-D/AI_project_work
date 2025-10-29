# 🐭 Mouse Social Behavior Recognition (MABe Challenge)

## 📌 Project Overview
This project is part of our AI Application course group work.  
We participate in the [MABe Challenge on Kaggle](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection), which focuses on **automatic recognition of mouse social and non-social behaviors** from pose estimation data.  
Our goal is to design and implement machine learning models that can classify over 30 behaviors in pairs and groups of mice, based on large-scale annotated video datasets.

---
## 👥 Team Members

| Name         | Major              | Contact                     |
|--------------|-------------------|-----------------------------|
| **CHEN JINQIU** | Information Systems | 15007499500@163.com         |
| **XU XIAQING** | Computer Science     | xiaqingxu623@gmail.com      |

## 🎯 Objectives
- Build baseline and advanced models for behavior recognition  
- Explore spatio-temporal and multi-modal feature representations  
- Submit valid predictions to Kaggle and evaluate performance using the official F-score metric  
- Document our methodology, experiments, and results for the course project  

---

## 📊 Dataset
- **Source**: MABe Challenge Dataset  
- **Scale**: 400+ hours of video, 20+ recording systems, 30+ behaviors, frame-level expert annotations  
- **Note**: Due to its size, the dataset is *not included* in this repository  
- **Access**: Download via Kaggle API or directly from the competition page  
- We will provide instructions in `docs/dataset_instructions.md` for setting up the dataset locally  

---

## 📂 Repository Structure
- `src/` : Core source code (data processing, model training, evaluation)  
- `notebooks/` : Jupyter notebooks for exploration and experiments  
- `docs/` : Project documentation, reports, and plans  
- `results/` : Visualizations, logs, and evaluation outputs  
- `submission/` : Kaggle submission files (submission.csv)  
- `README.md` : Project overview  

---

## ⚙️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/Luffy003-D/AI_project_work.git
cd AI_project_work

# Install dependencies
pip install -r requirements.txt

# Example: run training with default config
python src/train.py --config configs/default.yaml
