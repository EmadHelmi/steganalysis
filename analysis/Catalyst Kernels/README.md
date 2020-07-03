# An Improved CNN Steganalysis Architecture Based on “Catalyst Kernels” and Transfer Learning

## About the reference

This code is an implementation of the mentioned paper above in the title. You can also find that paper [here](https://link.springer.com/chapter/10.1007/978-3-319-97749-2_9).

## How to run the code

You can run the code as below:

```bash
python main.py --ctrp <clean train path> --ctep <clean test path> --strp <stego train path> --step <stego test path> --nc <number of classes> --ne <number of epochs> --bs <batch size> --assert_model <if you want to assert your model> --shuffle <if you want to shuffle your data> -v <to verbose the output of training>  --op <output path for saved models>
```

Also you can run `python main.py -h` to get more help