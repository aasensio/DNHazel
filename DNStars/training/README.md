Training without noise:
python train.py --action start --out cnns/test --epochs 10

Training without noise:
python train.py --action start --out cnns/test --epochs 10 --noise 1e-4

Prediction
python train.py --action predict --out cnns/test --noise 1e-4
