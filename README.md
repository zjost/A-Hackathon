# DGL Hackathon

Many real-world data contain relation information, for example, friend relation in social network. Such relation information between entities can be of great help to improve accuracy of models that only use entity feature. Relations are usually modeled with graphs. To better exploit information on such relation graphs, DGL is designed and is now the state-of-the-art framework for developing deep learning models on graphs. In this hackathon, participants will learn the basics of graphs and related applications, how deep learning is used in these graph applications, and how to use DGL to implement the graph models for some simple applications.

**Projects:**
DAVE LOVES REDDIT. He spends almost every second on it: browsing news, asking questions or simply meming. One headache Dave always has is to search for the right subreddit to post. God knows r/WtWFotMJaJtRAtCaB stands for “When the Water Flows over the Milk Jug at Just the Right Angle to Create a Bubble”. Dave **wants an algorithm that can predict the subreddit category of a reddit post**. In this hackathon, you need to help Dave to develop such an algorithm using Graph Neural Networks and DGL.

To bootstrap, we further divide the project into following sessions:

* DGL and community detection 101 in 30 minutes
    * We will go through a jupyter notebook with the participants together using a similar but much simpler example on Zachery’s karate club network. We hope this could give the audience some basic background.
* Milestone 1 (basic): Beat the MLP baseline
    * Following the example in the 101 session, the participants should try to apply a similar Graph Neural Network on the provided Reddit dataset and achieve better accuracy than a MLP baseline.
    * DGL already has many datasets and models integrated. We encourage participants to try different model or improve existing model for better accuracy. A leaderboard will display the current winner.
* Milestone 2 (advanced): Using raw Reddit data
    * The Reddit post graph we provided is only one of the many ways to model the raw Reddit dataset. Let’s see how can you further improve the accuracy by different approaches.
    * Potential direction #1: Try constructing the post-post graph using user id, timestamp, or others. One can also try out adding global super node, second-order neighbors, or other more advanced methods.
    * Potential direction #2: Extract node features with GluonNLP
        * Node features have a great impact of the performance of the model. In the raw Reddit dataset, text of the post is provided with each node and the problem here is how to extract more efficient features from these raw data. The participants can use knowledge they learned from GluonNLP to design models for generating better node features.
