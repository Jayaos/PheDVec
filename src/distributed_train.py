import tensorflow as tf
from src.phedvec import PhedVec
from src import utils

class Train(object):
    """Class for training PhedVec model
    Args:
        epochs: Number of epochs
        model: PhedVec model
        batch_size: Batch size
        strategy: Distribution strategy in use
    """

    def __init__(self, model, strategy, epochs, batch_size, lr=0.01):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_metric = [] # need revisit
        self.model = model
        self.strategy = strategy

    def compute_visitloss(self, labels, predictions):
        per_example_loss = self.loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)
    
    def compute_conceptloss(self, i_vec, j_vec):
        logEps = tf.constant(1e-5)
        norms = tf.reduce_sum(tf.math.exp(tf.matmul(self.model.embedding, self.model.embedding, transpose_b=True)), axis=1)
        denoms = tf.math.exp(tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(self.model.embedding, i_vec), 
                                                       tf.nn.embedding_lookup(self.model.embedding, j_vec)), axis=1))
        concept_loss = tf.negative(tf.math.log((tf.divide(denoms, tf.gather(norms, i_vec)) + logEps)))
        return tf.nn.compute_average_loss(concept_loss, global_batch_size=self.batch_size)

    def train_step(self, input_batch):
        """One train step.
        Args:
        inputs: one batch input
        Returns:
        loss: Scaled loss.
        """

        def pickij(visit_record, i_vec, j_vec):
            unpadded_record = visit_record[visit_record != 0]
            for first in unpadded_record:
                for second in unpadded_record:
                    if first == second: continue
                    i_vec.append(first)
                    j_vec.append(second)

        x_batch, labels = input_batch
        i_vec = []
        j_vec = []
        
        for x in x_batch:
            pickij(x, i_vec, j_vec)
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch)
            loss = self.compute_visitloss(labels, predictions) + self.compute_conceptloss(i_vec, j_vec)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def custom_training(self, train_dist_ds, total_len, strategy):
        """Custom distributed training loop
        """
        def distributed_train_epoch(ds, total_len, batch_size):
            total_loss = 0.0
            num_train_batches = 0.0
            training_len = total_len / batch_size
            progbar = tf.keras.utils.Progbar(training_len)
            for one_batch in ds:
                per_replica_loss = strategy.run(self.train_step, args=(one_batch,))
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                num_train_batches += 1
                progbar.add(1)
            return total_loss, num_train_batches
        
        for epoch in range(self.epochs):
            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_ds, total_len, self.batch_size)

            template = ('Epoch: {}, Train Loss: {}')
            print(template.format(epoch, train_total_loss / num_train_batches))

def main(epochs, buffer_size, batch_size, config_dir, num_gpus=2, lr=0.01):
    """main function for implementation"""

    devices = ['/device:GPU:{}'.format(i) for i in range(num_gpus)]
    strategy = tf.distribute.MirroredStrategy(devices)
    train_dataset, total_len = utils.create_dataset(buffer_size, batch_size, config_dir)

    with strategy.scope():
        phedvec = PhedVec(config_dir)
        Trainer = Train(phedvec, strategy, epochs, batch_size, lr)

        train_distributed_datset = strategy.experimental_distribute_dataset(train_dataset)

    print('Training...')
    Trainer.custom_training(train_distributed_datset, total_len, strategy)