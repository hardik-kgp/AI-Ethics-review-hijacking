# ai_ethics_group_8
Term project for AI Ethics course
## Dataset
code for creating the data:
[Drive](https://colab.research.google.com/drive/1HvKhgAEW4lKMCGt7kIGyvvLIEdVJWMAv?usp=sharing)

## Dataset link 

[Drive](https://drive.google.com/drive/folders/1Id-PmmKHSqgPjVO3lxaNJM48Vkd26Rdw)

## Notebook for Twin LSTM implementation 
Without emanuals - https://colab.research.google.com/drive/1kG2INMtvEuNgLKGc1qdY49WCPt5BW0pb?usp=sharing

With emanuals - https://colab.research.google.com/drive/1N-tTlm-cLhBXKKMxIqq3eTMVc9-8i1z3?usp=sharing

## Dataset statistics
|split|related|sbsc|sbdc|dbsc|dbdc|
|---|---|---|---|---|---|
|train|27.727|20.614|20.001|17.153|14.504|
|val|27.731|20.601|20.017|17.153|14.498|
|test|27.723|20.612|20.0|17.157|14.508|

## Performance comparison of models

|model|acc|macro f1|related acc|sbsc acc|sbdc acc|dbsc acc|dbdc acc|
|---|---|---|---|---|---|---|---|
|BERT|81.682|83.91|95.273|97.94|97.771|63.137|67.202|
|BERT e-manual product (back)|81.307|83.533|95.516|97.892|96.84|62.395|66.825|
|BERT e-manual product+review (back)|81.307|83.627|94.022|98.706|97.65|63.676|66.023|
|BERT e-manual review (back)|81.543|83.831|95.134|97.46|98.784|64.553|65.371|
|BERT-XD|81.3|83.519|95.169|97.508|98.298|60.371|67.854|
|BERT-XD e-manual product (back)|81.154|83.307|93.917|97.844|98.784|57.673|69.759|
|E-BERT|81.821|83.784|94.612|98.083|98.784|56.358|72.518|
|E-BERT e-manual product (back)|81.383|83.724|94.821|98.275|97.893|**68.465**|62.237|
|E-BERT e-manual product+review (back)|80.827|82.73|95.481|97.413|97.771|52.614|72.066|
|E-BERT e-manual review (back)|81.467|83.498|95.586|97.988|97.528|57.099|70.812|
|E-RoBERTa|81.557|83.868|95.377|97.508|98.622|64.486|65.371|
|E-RoBERTa e-manual product (back)|81.293|83.667|93.883|98.323|98.784|67.487|62.738|
|E-RoBERTa e-manual product+review (back)|80.918|83.28|94.195|98.227|98.015|60.641|66.775|
|E-RoBERTa e-manual review (back)|81.349|83.531|95.516|98.131|97.326|59.022|69.057|
|RoBERTa|**81.898**|83.949|94.3|98.563|98.825|56.155|**72.894**|
|RoBERTa e-manual product (back)|81.703|**84.054**|95.481|97.988|97.853|67.218|64.017|
|RoBERTa e-manual product+review (back)|81.648|83.649|**95.899**|**98.754**|97.366|54.739|72.693|
|RoBERTa e-manual review (back)|80.89|83.049|95.238|97.94|98.663|53.761|70.787|
|TwinLSTM|76.107|77.905|83.594|97.652|**98.987**|47.622|66.449|

<!-- ## Performance comparison of models
|model                   |acc       |macro f1  |related acc|sbsc acc  |sbdc acc  |dbsc acc  |dbdc acc  |
|------------------------|----------|----------|-----------|----------|----------|----------|----------|
|TwinLSTM                |76.107    |77.905    |83.594     |97.652    |**98.987**|47.622    |66.449    |
|BERT                    |81.682    |83.91     |95.273     |97.94     |97.771    |63.137    |67.202    |
|BERT (w prod-emanual)   |81.481    |83.439    |94.508     |98.323    |97.731    |56.459    |71.815    |
|BERT-XD                 |81.3      |83.519    |95.169     |97.508    |98.298    |60.371    |67.854    |
|BERT-XD (w prod-emanual)|81.154    |83.307    |93.917     |97.844    |98.784    |57.673    |69.759    |
|RoBERTa                 |**81.898**|**83.949**|94.3       |**98.563**|98.825    |56.155    |**72.894**|
|RoBERTa (w prod-emanual)|81.432    |83.76     |**95.377** |97.413    |98.217    |**64.418**|65.271    | -->
 
 **w prod-emanual**: emanual data added to product info.
 **BERT-XD**: Results obtained using the [BERT-XD_Review](https://huggingface.co/activebus/BERT-XD_Review) model.

<!-- ### General Statistics
1) 2276 data points (reviews). <br>
2) 47 products <br>
3) 1382 genuine reviews and 894 fake "hijacked reviews", maintains the original 34:22 ratio of the paper.
4) No examples of same brand, different sub category. <br>
5) All the other cases (_dbsc_: different brand, same category, _dbdc_: different brand, different category and _sbsc_: same brand, same category) are almost equal: <br>
**dbsc**: 256 <br> 
**dbdc**: 386 <br>
**sbsc**: 252 <br>
### Train-Test split
made using stratified sampling

**class distribution of splits** <br>

|  Split Type  | related | same brand, same category | diff brand, same category | diff brand, diff category |
|--------------|---------|---------------------------|---------------------------|---------------------------|
| `Train`      |   1105  |             201           |             205           |             309           |
| `Test`       |    277  |              51           |              51           |              77           |


**Initial BERT baseline results on this split:** <br>
best test accuracy: 72.59%

**class wise accuracy:** <br>
related: 96.76% <br> 
same brand, same category: 0% (literally can't detect it seems) <br> 
different brand, same category (easily detected): 100% <br> 
different brand, different category:  12.82% (also hard to detect, a bit weird) <br> -->

### NOTE:

<!-- rename ReviewBERT_pb to EManualBERT_pb -->
<!-- rename ReviewBERT_rb to EManualBERT_rb -->
rename ReviewBERT_prb to EManualBERT_prb