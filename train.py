import tensorflow as tf
import numpy as np
import tqdm, math

from models import regressionnet

def evaluate(net, pose_loss_op, test_iterator, summary_writer, tag="test/pose_loss"):
    test_it = copy.copy(test_iterator)
    total_loss = 0.0
    cnt = 0
    num_batches = int(math.ceil(len(test_it.dataset) / test_it.batch_size))
    print(len(test_it.dataset))
    for batch in tqdm(test_it, total=num_batches):
        feed_dict = regressionnet.fill_joint_feed_dict(net, regressionnet.batch2feeds(batch)[:3], conv_lr=0.0, fc_lr=0.0, phase='test')
        global_step, loss_value = net.sess.run([net.global_iter_counter, pose_loss_op], feed_dict=feed_dict)
        total_loss += loss_value * len(batch)
        cnt += len(batch)
    avg_loss = total_loss / len(test_it.dataset)
    print ('Step {} {} = {:.3f}'.format(global_step, tag, avg_loss))
    summary_writer.add_summary(create_sumamry(tag, avg_loss), global_step=global_step)

def train():
    return raise ValueError("train function will be updated soon")

def main(argv):
    return (argv)

if __name__ == "__main__":
    main(sys.argv[1:])
