### References

Literature review is almost done. Finalizing the code design and setup.

1. [https://jduarte.physics.ucsd.edu/phys139_239/finalprojects/Group3_Report.pdf](https://jduarte.physics.ucsd.edu/phys139_239/finalprojects/Group3_Report.pdf)
   1. One of the basic ways to implement a GNN for track reconstruction
   2. [https://github.com/gnn-tracking/gnn_tracking](https://github.com/gnn-tracking/gnn_tracking)
      1. This is their updated github repository
      2. Comes from an issue in the repo, related to better hyperparameter tuning of residual networks
         1. "Switch to using MLPs with residual connections in INs"
         2. IN refers to the interaction networks
         3. [https://arxiv.org/pdf/2309.16620](https://arxiv.org/pdf/2309.16620)
2. [https://arxiv.org/pdf/2012.01249](https://arxiv.org/pdf/2012.01249)
   1. 2020 paper covering end to end of using GNNs in reconstruction
3. [https://indico.cern.ch/event/1027582/contributions/4386630/attachments/2254420/3824974/GNNs%20%26%20Graph%20Techniques%20for%20Track%20Reconstruction%20(4).pdf](https://indico.cern.ch/event/1027582/contributions/4386630/attachments/2254420/3824974/GNNs%20%26%20Graph%20Techniques%20for%20Track%20Reconstruction%20(4).pdf)
   1. 2021 paper discussion about using Fast Fixed Radius Nearest Neighbours method for faster inference
4. [https://github.com/GageDeZoort/gnns-for-tracking](https://github.com/GageDeZoort/gnns-for-tracking)
   1. TrackML data for two models:
      1. Edge classification to predict hit associations
      2. Object condensation to cluster hits and predict track properties
5. [https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_02032.pdf](https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_02032.pdf)
    1. 2023 paper containing insights for better metric learning using the TrackML dataset
    2. Use of quantization for model resource optimization (Quantization Aware Training)
6. [https://indico.cern.ch/event/1104699/attachments/2446264/4196018/GNN%20for%20HL-LHC.pdf](https://indico.cern.ch/event/1104699/attachments/2446264/4196018/GNN%20for%20HL-LHC.pdf)
    1. 2022 seminar about using GNNs in reconstruction.
    2. Encompasses 3 and goes beyond
7. [https://arxiv.org/pdf/2203.12852](https://arxiv.org/pdf/2203.12852)
    1. 2022 paper discussing current situation, innovations and challenges in GNNs for track reconstruction
8. [https://cds.cern.ch/record/2918902/files/ATL-DAQ-SLIDE-2024-614.pdf](https://cds.cern.ch/record/2918902/files/ATL-DAQ-SLIDE-2024-614.pdf)
    1. 2024 paper discussing latest works, continuation of 3 and 6, but fast paced and high level
9. [https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2022.828666/full](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2022.828666/full)
    1. 2022 paper discussing throughput and resource optimized methods for GNNs
10. [https://github.com/ryanliu30/HierarchicalGNN/tree/main](https://github.com/ryanliu30/HierarchicalGNN/tree/main)
    1. Code + paper saying hierarchical GNNs are better than flat GNNs
11. [https://arxiv.org/pdf/2106.10866v3](https://arxiv.org/pdf/2106.10866v3)
    1. GNN reweighting
