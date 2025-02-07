# CNN Training Indexer  

This project modifies a Convolutional Neural Network (CNN) to output a **feature vector** instead of a probability distribution.  

## 🔍 Why do this?  

By extracting and saving feature vectors from the training dataset, you can efficiently perform **cosine similarity search** to retrieve the most similar image in the dataset—along with its label.  

Essentially, this transforms the CNN into an **indexing model** that tells you:  
1. **What** the model predicts.  
2. **Where** the model learned that information.  

### 📌 Experiment Inspiration  
This idea is inspired by:  
[Lloyd Watts | Solving The Billion-Dollar Problems in AI: LLM Explainability/Hallucinations and More](https://www.youtube.com/watch?v=ptONcdI9ggA)  

## 📊 Example Results  

We trained our CNN on the **MNIST dataset**, and here’s what the results look like:  

![image](https://github.com/user-attachments/assets/9b8c1a65-cb11-468f-99d1-a5ab120b2d80)  
![image](https://github.com/user-attachments/assets/6976e4c9-06cb-473f-8d39-6672880ee787)  
![image](https://github.com/user-attachments/assets/1b7dc66a-fe37-481d-a0d0-ea2437ad44c2)  
