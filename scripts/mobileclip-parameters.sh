# MobileCLIP2
for model in S0 S2 B S3 L-14 S4
do
  hf download apple/MobileCLIP2-$model
done

# MobileCLIP
for model in S0 S1 S2 B B-LT S3 L-14 S4
do
  hf download apple/MobileCLIP-$model
done
