# MCG
Multi-level Contrastive Graph Learning for Academic Abnormality Prediction

The model code cites the literature when generating virtual nodes for the imbalance alleviation problem: 'GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks' https://github.com/ TianxiangZhao/GraphSmote.

The paper Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning is cited in the context of multi-level contrastive learning for mitigating long-tailed distribution problems. https://github.com/ GRAND-Lab/MERIT

The model training requires the student's performance information and behavior association matrix as input.

In the construction of behavioral associations, students living in the same dormitory are constructed as dormitory behavioral associations, and students choosing the same elective course are constructed as interest behavioral associations, and then the above two behavioral associations are fused to obtain the final behavioral adjacency matrix.
