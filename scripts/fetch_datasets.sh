#!/usr/bin/env bash
set -e
OUTDIR="datasets"

# retrieve the datasets
curl -o ${OUTDIR}/zebf-auditory-restoration-1.zip https://figshare.com/ndownloader/files/55083911

# unpack the datasets
for zipfile in ${OUTDIR}/*.zip; do
    if [ -f "$zipfile" ] ; then
	unzip -o ${zipfile} -d ${OUTDIR}
    fi
done

# delete the zip files
rm ${OUTDIR}/*.zip