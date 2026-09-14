param([string]$Config = "configs\smoke.json")
python scripts\evaluate.py --config $Config --split test

