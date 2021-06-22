To run the training script,

python Model_Train.py \
--freeze-backbone \
--random-transform \
--epochs 30 \
--batch-size 8 \
csv train_annotations.csv classes.csv \
--val-annotations val_annotations.csv

train_annotations.csv and val_annotations.csv have to be in the format,
<Img_path>, <x1>, <y1>, <x2>, <y2>, <Class>

classes.csv has to be in the format,
<Class_Name>, <Mapping_to_Int>
E.g. Car, 0

To run the prediction script,
python predicting_labels.py \
--rootpath {Path where you want to save the prediction file} \
--testpath {Path for test set directory} \
--modelpath {Path for model weights to load the model}