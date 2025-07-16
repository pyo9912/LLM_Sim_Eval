# Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models

This repo provides the source code & data based on the paper: [Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models](https://arxiv.org/abs/2305.13112) (Arxiv 2023).

## 😀 Overview for the base project

**Highlights**:
- 1️⃣ We are the first to examine ChatGPT in conversational recommendation systematically, the ability of which is **underestimated** in traditional evaluation approach.
- 💡 We propose a new interactive approach that employs LLM-based user simulators for evaluating CRSs.
- 🔝 The recall@50 metric can be boosted from 0.218 to 0.739 on the redial dataset with our new interactive evaluation approach, even surpassing the currently leading CRS baseline.

we propose an **i**nteractive **Eval**uation approach based on **LLM**s named **iEvaLM** that harnesses LLM-based user simulators. We take the ground-truth items from the example as the user preference through the interaction, and use them to set up the persona of the simulated users by LLMs through instructions. To further make a comprehensive evaluation, we consider two types of interaction: *attribute-based question answering* and *free-form chit-chat*.

<p align="center">
  <img src="./asset/eval.png" width="75%" height="75% title="Overview of iEvaLM-CRS" alt="">
</p>


## 🌟 Perfermance
**Highlights**:
- 🔝 In this project, we evaluated recommendation accuracy using the framework proposed by iEvaLM. The results are as follows.
- 🦙 Additionally, we included the performance of the LLaMa3-8b-Instruct model, which was not covered in iEvaLM.
<p align="center">Performance of CRSs and ChatGPT using different evaluation approaches.</p>
<table border="1" align="center">
  <tbody >
  <tr align="center">
    <td colspan="2">Model</td>
    <td colspan="3">KBRD</td>
    <td colspan="3">UniCRS</td>
    <td colspan="3">LLaMa3</td>
    <td colspan="3">ChatGPT</td>
  </tr>
  <tr align="center">
    <td rowspan="3">ReDial</td>
    <td>H@1</td>
    <td colspan="3">0.030</td>
    <td colspan="3">0.110</td>
    <td colspan="3">0.389</td>
    <td colspan="3">0.435</td>
  </tr>
  <tr align="center">
    <td>H@5</td>
    <td colspan="3">0.113</td>
    <td colspan="3">0.298</td>
    <td colspan="3">0.592</td>
    <td colspan="3">0.722</td>
  </tr>
  <tr align="center">
    <td>H@10</td>
    <td colspan="3">0.197</td>
    <td colspan="3">0.423</td>
    <td colspan="3">0.665</td>
    <td colspan="3">0.806</td>
  </tr>
  </tbody>

</table>

                      
## 🐦 Citing

Please cite the following paper if you find our code helpful.

```bibtex
@article{wang2023rethinking,
  title={Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models},
  author={Wang, Xiaolei and Tang, Xinyu and Zhao, Wayne Xin and Wang, Jingyuan and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2305.13112},
  year={2023}
}
```
