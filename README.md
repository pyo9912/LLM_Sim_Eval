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


## 🚀 Quick Start

### Requirements

- python == 3.9.16
- pytorch == 1.13.1
- transformers == 4.28.1
- pyg == 2.3.0
- accelerate == 0.18.0

### Download Models

You can download our fine-tuned models from the [link](https://drive.google.com/drive/folders/1h2AcRn3cd9qXToM9hFAXFV5evXJM-wyD?usp=sharing), which include recommendation and conversation models of **KBRD**, **BARCOR** and **UniCRS**. Please put the downloaded model into src/utils/model directory.

### Interact with the user simulator

- dataset: [redial, opendialkg]
- mode: [ask, chat]
- model: [kbrd, barcor, unicrs, chatgpt]

```bash
cd script
bash {dataset}/cache_item.sh 
bash {dataset}/{mode}_{model}.sh 
```

You can customize your iEvaLM-CRS by specifying these configs:
- `--api_key`: your API key
- `--turn_num`: number of conversation turns. We employ five-round interaction in iEvaLM-CRS.

After the execution, you will find detailed interaction information under "save_{turn_num}/{mode}/{model}/{dataset}/".

### Evaluate

```bash
cd script
bash {dataset}/Rec_eval.sh
```

You can customize your iEvaLM-CRS by specifying these configs:
- `--turn_num`: number of conversation turns.
- `--mode`: [ask, chat]

After the execution, you will find evaluation results under "save_{turn_num}/result/{mode}/{model}/{dataset}.json".


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
    <td>0.030</td>
    <td>0.110</td>
    <td>0.389</td>
    <td>0.435</td>
  </tr>
  <tr align="center">
    <td>H@5</td>
    <td>0.113</td>
    <td>0.298</td>
    <td>0.592</td>
    <td>0.722</td>
  </tr>
  <tr align="center">
    <td>H@10</td>
    <td>0.197</td>
    <td>0.423</td>
    <td>0.665</td>
    <td>0.806</td>
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
