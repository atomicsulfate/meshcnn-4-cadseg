from meshcnn.options.test_options import TestOptions
from meshcnn.data import DataLoader
from models import create_model
from meshcnn.util.writer import Writer

def create_test_options():
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    return opt

def test_model(model, opt, epoch):
    dataset = DataLoader(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc

def run_test(epoch=-1):
    print('Running Test')
    opt = create_test_options()
    model = create_model(opt, 0)
    return test_model(model, opt, epoch)

def run_validation(model, epoch):
    print('Running Validation')
    model.set_train(False)
    acc = test_model(model, create_test_options(), epoch)
    model.set_train(True)
    return acc


if __name__ == '__main__':
    run_test()
