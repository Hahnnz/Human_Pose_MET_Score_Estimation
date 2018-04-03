import tensorflow as tf
import numpy as np
import tqdm, math, sys, copy, cmd_options, datetime

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

def train(net, saver, loss_op, pose_loss_op, train_op, train_iterator, test_iterator, val_iterator, 
          max_iter=None, test_step=None, snapshot_step=None, log_step=1, batch_size=None, conv_lr=None, fc_lr=None, fix_conv_iter=None, output_dir="results"):
    with net.graph.as_default():
        summary_writer = tf.summary.FileWriter(output_dir, net.sess.graph)
        summary_op = tf.summary.merge_all()
        fc_train_op = net.graph.get_operation_by_name("fc_train_op")
    global_step = None
    
    for step in range(max_iter+1):
        # Test 
        if step % test_step ==0 or step +1 == max_iter or step == fix_conv_iter:
            global_step = net.sess.run(net.global_iter_counter)
            regressionnet.evaluate_pcp(net, pose_loss_op, test_iterator, summary_writer)
            if val_iterator is not None:
                regressionnet.evaluate_pcp(net, pose_loss_op, val_iterator, summary_writer) 
        # Snapshot
        if step % snapshot_step == 0 and step > 1:
            checkpoint_prefix = os.path.joint(output_dir, "checkpoint")
            assert global_step is not None
            saver.save(net.sess, checkpoint_prefix, global_step=global_step)
        if step == max_iter: break
        
        # Train
        feed_dict = regressionnet.fill_joint_feed_dict(net, 
                                                       regressionnet.batch2feeds(train_iterator.next())[:3],
                                                       conv_lr=conv_lr, fc_lr=fc_lr, phase='train')
        if step < fix_conv_iter:
            feed_dict["lr/conv_lr:0"]=0.0
            cur_train_op = fc_train_op
        else: cur_train_op = train_op
        
        if step % summary_step ==0:
            global_step, summary_str, _, loss_value = net.sess.run(
                [net.grobal_iter_counter, summary_op, cur_train_op, pose_loss_op],
                feed.dict=feed.dict)
            summary_writer.add_summary(summary_str, global_step=global_step)
        else:
            global_step, summary_str, _, loss_value = net.sess.run(
                [net.grobal_iter_counter, summary_op, cur_train_op, pose_loss_op],
                feed.dict=feed.dict)
        if step % log_step ==0 or step +1 == max_iter:
            print("Step %d: train/pose_loss = %.2f." % (global_step, loss_value))
def main(argv):
    
    args = cmd_options.get_arguments(argv)

    if not os.path.exists(args.o_dir): os.makedirs(args.o_dir)
    
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    
if __name__ == "__main__":
    main(sys.argv[1:])