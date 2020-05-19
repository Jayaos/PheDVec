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

    def __init__(self, model, epochs, batch_size, lr=0.01, strategy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
        self.loss_metric = [] # need revisit
        self.model = model
        self.strategy = strategy

    def compute_loss(self, labels, predictions):
        per_example_loss = self.model.compute_cost(x, labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, self.batch_size)

    def train_step(self, input_batch):
        """One train step.
        Args:
        inputs: one batch input
        Returns:
        loss: Scaled loss.
        """

        x_batch, label = input_batch
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss = self.compute_loss(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def custom_training(self, train_dist_ds, strategy):
        """Custom distributed training loop
        """

        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in ds:
                per_replica_loss = strategy.run(self.train_step, args=(one_batch,))
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                num_train_batches += 1
            return total_loss, num_train_batches
        
        for epoch in range(self.epochs):
            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_ds)

            template = ('Epoch: {}, Train Loss: {}')
            print(template.format(epoch, train_total_loss / num_train_batches))

def main(epochs, buffer_size, batch_size, lr=0.01, data_dir, config_dir, num_gpus=2):
    """main function for implementation"""

    devices = ['/device:GPU:{}'.format(i) for i in range(num_gpus)]
    strategy = tf.distribute.MirroredStrategy(devices)
    train_dataset = utils.create_dataset(buffer_size, batch_size, data_dir)

    with strategy.scope():
        phedvec = PhedVec(config_dir)
        Trainer = Train(phedvec, epochs, batch_size, lr, strategy)

        train_distributed_datset = strategy.experimental_distribute_dataset(train_dataset)

    print('Training...')
    Trainer.custom_training(train_distributed_datset, strategy)