# GCL-Formula-Retrieval
## Setup
### Packages
Test under Python 3.10.10 in Ubuntu. Install the required packages by
```
$ pip install -r requirements.txt
```
### Evaluaion tool
Evalution by [trec_eval](https://github.com/usnistgov/trec_eval). Insatll the tool under path Retrieval_result/.

## File Description
The following files are under datasets/
* download data: [ecir-2020](https://drive.google.com/drive/folders/1emboT7k4m7yKjru3AOb1xScZgbUnQuC8).
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
```
$ cd datasets/encoder
$ tar zxvf opt_list.txt.tzg
$ tar zxvf slt_list.txt.tzg
```
### Train
select <train_model>:  train_InfoGraph.py, &nbsp; train_GCL.py, &nbsp; train_BGRL.py
* Usage:
  ```
  $ python <train_model>
    --encode <slt or opt>
    --bs <batch size>
    --pretrained <set to use Tangent-CFT embedding as feature>
    --run_id <run id>
  ```
* example:
  ```
  $ python train_InfoGraph.py --encode opt --bs 2048 --pretrained --run_id 1
  ```
### Combine SLT and OPT
select <test_model>: test_InfoGraph.py,&nbsp;  test_GCL.py,&nbsp;  test_BGRL.py
* usage:
  ```
  $ python <test_model>
    --bs <batch size>
    --pretrained <set to use Tangent-CFT embedding as feature>
    --run_id <run id>
  ```
* example:
  ```
  $ python test_InfoGraph.py --bs 2048 --pretrained --run_id 1
  ```
### Evaluation
* Above retrieval result file will be saved at:
```
Retrieval_result/<model>/<graph encode form>/<batch size>/<run id>/<retrieval_res>
```
* usage:
```
$ cd Retrieval_result/
$ ./trec_eval/trec_eval ./NTCIR12_MathWiki-qrels_judge.dat <retrieval file path> | grep bpref
```
* example:
```
$ cd Retrieval_result/
$ ./trec_eval/trec_eval NTCIR12_MathWiki-qrels_judge.dat GCL/opt/2048/1/retrieval_res5_1_end | grep bpref
```
