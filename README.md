# supernet-two-stages
Anonymized code for NeurIPS submission under review, titled "Understanding Architecture Selection in Supernet Neural Architecture Search".
The paper consists of a case study and main experimental portion, which are implemented separately in code.

## Case study
The case study implements two DARTS variants, DARTS-PT and DARTS-, together to compare the interaction of their modifications to the algorithm.
To achieve this, we modified the published code of [DARTS-PT](https://github.com/ruocwang/darts-pt) to implement DARTS-.
The case study directory [provides usage instructions](https://github.com/anon-submit-ml/picking-up-pieces/tree/main/case_study) for both the original implementation and our extension.

## Two Stage Supernet Search experiments

The code used for the main experimental portion was based on the [published implementation of DrNAS.](https://github.com/xiangning-chen/DrNAS)
We have extended the original code, implementing DARTS- and RSPS in the NAS-Bench-201 search space, with random search with parameter sharing (RSPS) being used to design our baseline model.
Checkpoint files for our baseline model are provided in the [baselines subdirectory](https://github.com/anon-submit-ml/picking-up-pieces/tree/main/two_stage/baselines/nasbench201) so our Stage-2 search statistics can be recomputed or computed for additional Stage-2 search algorithms using the same baseline models.
The two-stage directory also contains our implementations of Stage-2 search algorithms, for which we have attempted to make as much use of the [original published](https://github.com/SamsungLabs/zero-cost-nas) [implementations](https://github.com/ruocwang/darts-pt) as possible.

Full usage instructions are provided [within the directory](https://github.com/anon-submit-ml/picking-up-pieces/blob/main/two_stage/README.MD) .
