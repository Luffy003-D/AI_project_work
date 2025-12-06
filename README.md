# 🐭 ITE 351 Group 2: Decoding the Secret Language of Mice

> **Project:** MABe Challenge - Social Action Recognition
> **Course:** ITE 351: AI & Applications (2025)
> **Status:** Final Submission

---

## 📋 Table of Contents
1. [The "Why": Beyond Just Mice](#1-the-why-beyond-just-mice)
2. [The Challenge: 400 Hours of Silent Movies](#2-the-challenge-400-hours-of-silent-movies)
3. [Our Solution: The "Translator" Approach](#3-our-solution-the-translator-approach)
4. [Technical Deep Dive](#4-technical-deep-dive)
5. [Key Challenges & Optimizations](#5-key-challenges--optimizations)
6. [Video Presentation](#6-video-presentation)

---

## 👥 Team Members

| Name         | Role                          | Department          | Email                  |
| :----------- | :---------------------------- | :------------------ | :--------------------- |
| *XU,XIAQING* | Model Architect&Data Engineer | Computer Science    | xiaqingxu623@gmail.com |
| CHEN,JINQIU  | Documentation&Optimization    | Information Systems | 15007499500@163.com    |

---

## 1. The "Why": Beyond Just Mice 🧠

Why do scientists care if a mouse is "sniffing" or "chasing" another mouse?

It sounds trivial, but it is **fundamental to neuroscience**. Mice are the standard model for studying human brain disorders like autism, depression, and schizophrenia. These disorders often manifest specifically as changes in *social behavior*.

> **The Problem:** Traditionally, researchers have to watch thousands of hours of video, manually clicking: *"Frame 100: Mouse A sniffs Mouse B."* It is slow, subjective, and painful.

**Our Goal:** To build an AI that watches these videos and **automatically documents** the social life of mice, accelerating drug discovery and brain research.

---

## 2. The Challenge: 400 Hours of Silent Movies 🎥

The **MABe (Multi-Agent Behavior) Challenge** on Kaggle presents a unique difficulty. We don't get the actual video pixels; we only get **Pose Estimation Data** (Skeleton Keypoints).

Imagine watching a movie, but the screen is black, and you can only see moving white dots representing the actors' joints. Based *only* on those dots, you have to guess if they are fighting, hugging, or dancing.

### 📉 The Constraints
* **Input:** Coordinates of body parts (Nose, Ears, Tail base...) over time.
* **Output:** 30+ different behaviors (Sniff, Attack, Mount, etc.).
* **The Hard Part:**
    1.  **Data Chaos:** Data comes from different labs using different cameras.
    2.  **Time Limit:** The code must run in under **9 hours** on Kaggle (No Internet allowed).

---

## 3. Our Solution: The "Translator" Approach 🗣️

We treat this problem like a **Language Translation** task.
* **Input:** A sequence of movements (Body Language).
* **Output:** A sequence of behavior labels (Meaning).

We built an end-to-end Deep Learning pipeline:

1.  **Unified Data Loader:** Handling messy files (`.npy`, `.parquet`) from different labs.
2.  **Feature Engineering:** Giving the model "physics sense" (Velocity, Distance).
3.  **Transformer Model:** The same architecture behind ChatGPT, but for movement.

---

## 4. Technical Deep Dive 🛠️

### 4.1 Feature Engineering: Giving the Model "Eyes"
Raw coordinates (x, y)are boring. A mouse standing at (0,0) looks the same to a computer as one at (100,100). To fix this, we implemented a custom `FeatureExtractor` class.

We calculate:
* **Kinematics:** How fast is the nose moving? (Velocity & Acceleration)
* **Social Physics:** What is the distance between Mouse A's nose and Mouse B's tail?
    * *Intuition:* High velocity + Zero distance = `Attack` or `Chase`. Low velocity + Zero distance = `Sniff` or `Groom`.

### 4.2 The Model Architecture (Transformer)
We used a **Transformer Encoder**. Why not a simple RNN? Because behavior depends on context.

* **Window Size:** We look at **60 frames** (approx. 2 seconds) at once.
* **Self-Attention:** This mechanism allows the model to decide which frames are important. For example, the split second a mouse lunges is more important than the 10 frames of standing still before it.

```python
# Core logic from our 'TransformerClassifier' class
self.transformer_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=192, nhead=6, ...),
    num_layers=3
)
```

## 5. Key Challenges & Optimizations 🚀

The hardest part wasn't the AI—it was the **Engineering Constraints**.

### Challenge 1: "Out of Memory" (OOM) 💥

Video data is huge. Loading everything onto the GPU caused crashes.

- **Solution: Gradient Accumulation.**
- *How it works:* Think of it like cleaning a messy room. Instead of carrying all the trash out at once (which you can't carry), you fill small bags one by one, and throw them out together at the end. In code, we accumulate math over 8 small steps and update the model once.

### Challenge 2: The 9-Hour Limit ⏳

We had to be fast.

- **Solution:** We optimized our `MABeDataLoader` to efficiently parse heterogeneous file formats and pre-calculate features, stripping away any heavy visualization code during the training phase.

------

## 6. Video Presentation 📺

Check out our demo and explanation video below:

**[> Click Here to Watch Our Presentation <](https://www.google.com/search?q=%23&authuser=1)** *(Replace this with your YouTube link)*

------

*This blog was generated for the ITE 351 Group 2 Project.*
