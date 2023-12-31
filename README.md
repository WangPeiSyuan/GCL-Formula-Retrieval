# GCL-Formula-Retrieval

## Setup

### Packages
Tested under Conda 4.13.0 (Python 3.10.10) and [Conda 4.12.0](https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh) (Python 3.9.12) in Ubuntu. <br>
Create conda environment and install the required packages by running the following command:
```
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
$ pip install dgl==1.0.1+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
$ pip install PyGCL
$ pip install scipy==1.10
```
### Evaluaion tool
Evaluation is performed using [trec_eval](https://github.com/usnistgov/trec_eval). Install the tool in the "Retrieval_result/" directory.

## File Description
The following files are located under the "datasets/" directory:
* Download data: Access the data by following this [ecir-2020](https://drive.google.com/drive/folders/1emboT7k4m7yKjru3AOb1xScZgbUnQuC8) link.
* encoder/
  * opt_char_embeding.txt: Feature embedding with Tangent-CFT in OPT form
  * slt_char_embeding.txt: Feature embedding with Tangent-CFT in SLT form
  * opt_list.txt: Formula path in OPT form
  * slt_list.txt: Formula path in SLT form
  * query_opt_list.txt: Query formula path in OPT form
  * query_slt_list.txt: Query formula path in SLT form
  * opt_judge: Judged formula path in OPT form
  * slt_judge: Judged formala path in SLT form

## Quick Start
### Unzip file

Navigate to the "datasets/encoder" directory and unzip the files:
```
$ cd datasets/encoder
$ tar zxvf opt_list.txt.tgz
$ tar zxvf slt_list.txt.tgz
```

### Training using SLT or OPT encoding alone
Choose one of the following <train_model> options: "train_query_InfoGraph_slt_or_opt.py", "train_query_GCL_slt_or_opt.py", or "train_query_BGRL_slt_or_opt.py".
* Usage:
  ```
  $ python <train_model>
    --encode <slt or opt>
    --bs <batch size>
    --pretrained <set to use Tangent-CFT embedding as feature>
    --run_id <run id>
  ```
* Example:
  ```
  $ python train_query_InfoGraph_slt_or_opt.py --encode opt --bs 256 --pretrained --run_id 1
  ```

### Training using both SLT and OPT encodings

This script assumes that both the slt embedding and opt embedding are generated.

Choose one of the following <train_model> options: "train_query_InfoGraph_slt_plus_opt.py", "train_query_GCL_slt_plus_opt.py", or "train_query_BGRL_slt_plus_opt.py".

* Usage:
  ```
  $ python <train_model>
    --bs <batch size>
    --pretrained <set to use Tangent-CFT embedding as feature>
    --run_id <run id>
  ```
* Example:
  ```
  $ python train_query_InfoGraph_slt_plus_opt.py --bs 256 --pretrained --run_id 1
  ```

### Evaluation
* The above retrieval result file are saved in the following format:
  ```
  Retrieval_result/<model>/<graph encode form>/<batch size>/<run id>/<retrieval_res>
  ```
* To perform the evaluation, follow these steps:
  ```
  $ cd Retrieval_result/
  ```
 Choose one of the following measure options:
 "bpref" or "ndcg"
 * Usage:
   ```
   $ ./trec_eval/trec_eval -m <measure> ./NTCIR12_MathWiki-qrels_judge.dat <retrieval file path>
   ```
 * Example:
   ```
   $ ./trec_eval/trec_eval -m bpref ./NTCIR12_MathWiki-qrels_judge.dat GCL/opt/2048/1/retrieval_res5_1_end
   ```
 * For bpref full relevent:
   ```
   $ ./trec_eval/trec_eval -m bpref -l3 ./NTCIR12_MathWiki-qrels_judge.dat <retrieval file path>
   ```
 * Example:
   ```
   $ ./trec_eval/trec_eval -m bpref -l3 ./NTCIR12_MathWiki-qrels_judge.dat GCL/opt/2048/1/retrieval_res5_1_end 
   ```
  
