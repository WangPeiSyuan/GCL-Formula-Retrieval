# GCL-Formula-Retrieval

## Setup

### Packages
Tested under Python 3.10.10 in Ubuntu. Install the required packages by running the following command:
```
$ pip install -r requirements.txt
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

### Train
Choose one of the following <train_model> options: "train_InfoGraph.py", "train_GCL.py", or "train_BGRL.py".
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
  $ python train_InfoGraph.py --encode opt --bs 2048 --pretrained --run_id 1
  ```

### Combine SLT and OPT

This script assumes that both the slt embedding and opt embedding are generated.

Choose one of the following <test_model> options: "test_InfoGraph.py", "test_GCL.py", or "test_BGRL.py".

* Usage:
  ```
  $ python <test_model>
    --bs <batch size>
    --pretrained <set to use Tangent-CFT embedding as feature>
    --run_id <run id>
  ```
* Example:
  ```
  $ python test_InfoGraph.py --bs 2048 --pretrained --run_id 1
  ```

### Evaluation
* The above retrieval result file are saved in the following format:
  ```
  Retrieval_result/<model>/<graph encode form>/<batch size>/<run id>/<retrieval_res>
  ```
* To perform the evaluation, follow these steps:
  ```
  $ cd Retrieval_result/
  $ ./trec_eval/trec_eval ./NTCIR12_MathWiki-qrels_judge.dat <retrieval file path> | grep bpref
  ```
* Example:
  ```
  $ cd Retrieval_result/
  $ ./trec_eval/trec_eval NTCIR12_MathWiki-qrels_judge.dat GCL/opt/2048/1/retrieval_res5_1_end | grep bpref
  ```
