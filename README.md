### Computer Vision : Using SVM to recognize human facial expressions

#### 1. Requirements

* 1.Basic skills of Python programming
* 2.Basic understanding of SVM
* 3.dlib (for detecting facial landmarks)
* 4.scikit-image(for image preprocessing)
* 5.scikit-learn(for machine learning task)
* 6.numpy(for numeric processing)
* 7.matplotlib(for display image and plot data)
* 8.pickle(for saving Python's objects)
* 9.Jupyter notebook
#### 2. Introduction
This project is an demonstration of how we can make an intelligent system that reads an image of human face, then outputs their facial expressions.

There are 7 labels corresponding to 7 expressions, including  (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

Dataset can be downloaded from [https://www.kaggle.com/deadskull7/fer2013](https://www.kaggle.com/deadskull7/fer2013). This dataset consists of more than 30,000 images of different human's faces with several expressions

#### 3. Explaination
The first step in every machine learning project is to collect data for training and testing our model. Here we had got the dataset. The next things we need are features for out model to learn from.

So, what features can we use ?

In many machine learning projects, especially in the field of computer vision, people might use images's raw pixels as features. For examples, in hand-written digit classification problem(using MNIST dataset), using raw pixels is fine. But facial expression is not as much simple like that. There are some reasons:

* The first thing is if we use raw pixels, there will be a lot of features.
* Some features won't be useful, because they have low variants, or they do not represent any real-life logic(e.g your hair is black doesn't mean you are happy).

Those lead to a need of more useful features. Fortunately, there are already a number of approaches which we can use. In this project, facial landmarks is choosen.

> Facial landmarks are special points on human's face. As you might guess, they look like bellow:
![Image not fount](https://avatars1.githubusercontent.com/u/13412929?s=200&v=4)

When you smile, the landmarks's locations will change, and we can use those locations(x,y) as features.

We now can use dlib to detect facial landmarks. Thanks to it, we do not need to implement detecting algorithm from the ground.

After we have extracted features from the landmarks, we need to split data set into two part (training-testing set) with a reasonable ratio , in this example, test set will have size ratio equal to `0.1`.

The next step is to feed data to learning model (SVC in this case). We use Stochatic Gradient Desent to speed up training process. You can use Gridsearch to find the best hyper-parameters. Then use pickle to save our trained model to disk so we can re-use it next times.

#### 4. Conclusion

As expected, using Deep learning for this kind of task is a better choice (it archives more than 74% accuracy). There are some reasons :

* SVM is underfitting the dataset
* Human facial expression is not a simple topic. You might even cry when you feel happy. So there could be many mis-labeled training instances.

#### 5. Running the program

```python 
python ./extract_data_from_cvs.py
```

Launch Jupyter notebook then open `FacialExpressionSVM.ipynb`, run it.