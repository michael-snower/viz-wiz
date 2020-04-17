# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'ann' 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    mkdir -p $DOWNLOAD/$FOLDER
done

BLOB='https://convaisharables.blob.core.windows.net/uniter'

# annotations
NLVR='https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data'
wget -c $NLVR/dev.json -P $DOWNLOAD/ann/
wget -c $NLVR/test1.json -P $DOWNLOAD/ann/

# image dbs
for SPLIT in 'train' 'dev' 'test'; do
    wget -c $BLOB/img_db/nlvr2_$SPLIT.tar -P $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/nlvr2_$SPLIT.tar -C $DOWNLOAD/img_db
done

# test dbs
for SPLIT in 'train' 'dev' 'test1'; do
    wget -c $BLOB/txt_db/nlvr2_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/nlvr2_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

wget -c $BLOB/pretrained/uniter-base.pt -P $DOWNLOAD/pretrained/

wget -c $BLOB/finetune/nlvr-base.tar -P $DOWNLOAD/finetune/
tar -xvf $DOWNLOAD/finetune/nlvr-base.tar -C $DOWNLOAD/finetune
