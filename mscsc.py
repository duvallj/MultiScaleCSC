# Algorithm adapted from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5522776/
import numpy as np
import tensorflow as tf
import itertools


class FilterBlock(tf.keras.Model):
    def __init__(self, filter_scale, num_filters, in_channels=3, stride=1):
        super().__init__()

        self.filter_scale = filter_scale
        self.num_filters = num_filters
        initializer = tf.keras.initializers.GlorotUniform()
        # We are actually interested in reconstructing 
        shape = (filter_scale, filter_scale, num_filters, in_channels)
        self.filters = tf.Variable(
            initializer(
                shape=shape
            ),
            shape=shape,
            trainable=True
        )
        self.stride = (1, stride, stride, 1)

    def call(self, x, training=True):
        return tf.nn.conv2d(x, self.filters, self.stride, padding='SAME')

# TODO: come up with a better name for this
class ZBlock(tf.keras.Model):
    """
    Args:
      filter_blocks: list of FilterBlock objects
      height: height of images to recreate
      width: width of images to recreate
      alpha: regularization parameter
    """
    def __init__(self, filter_blocks, height, width, alpha, channels=3):
        super().__init__()

        self.filter_blocks = filter_blocks
        self.height = height
        self.width = width
        self.alpha = alpha
        self._channels = channels
        self._output_shape = (1, height, width, channels)
        self.z = []
        initializer = tf.keras.initializers.GlorotUniform()

        for filter_block in filter_blocks:
            shape = (1, height, width, filter_block.num_filters)
            z_s = tf.Variable(
                initializer(
                    shape=shape
                ),
                shape=shape,
                trainable=True
            )
            self.z.append(z_s)

    def recreate_one_image(self, x, training=True):
        reconstruction = tf.zeros(self._output_shape)
        for filter_block, z_s in zip(self.filter_blocks, self.z):
            reconstruction += filter_block(z_s, training=training)

        reconstruction = tf.reshape(reconstruction, self._output_shape[1:])
        return reconstruction

    def call(self, x, training=True):
        """
        Calculates the loss of reconstructing the batch of images given the
        current variables stored in self.z
        """
        reconstruction = tf.map_fn(
            fn=lambda i: self.recreate_one_image(i, training=training),
            elems=x
        )

        reconstruction_loss = tf.norm(x - reconstruction)
        encoding_loss = self.alpha * sum(tf.norm(z_s, ord=1) for z_s in self.z)

        return reconstruction_loss + encoding_loss

class Train:
    def __init__(self, epochs, z_epochs, batch_size, height, width,
                 num_channels, scales, d_lr, z_lr, alpha):
        """
        Args:
          epochs: The number of training epochs to run
          z_epochs: the number of z training epochs to run per d epoch
          batch_size: The image batch size to use
          scales: Array with elements (kernel_size, num_classes)
          d_lr: Initial learning rate to apply when training kernels
          z_lr: Initial learning rate to apply when recreating image from kernels
          alpha: regularization parameter to make sure kernel weights are never too large
        """
        self.epochs = epochs
        self.z_epochs = z_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.d_lr = d_lr
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=d_lr)
        self.z_opt = tf.keras.optimizers.Adam(learning_rate=z_lr)
        self.loss_metric = tf.keras.metrics.Sum(name='loss')

        self.filter_blocks = []
        for scale, num_classes in scales:
            self.filter_blocks.append(
                FilterBlock(scale, num_classes, in_channels=num_channels))

        self.zblock = ZBlock(self.filter_blocks, height, width, alpha)

    def decay(self, epoch):
        if epoch <= 10:
            return self.d_lr
        elif epoch <= 50:
            return 0.1 * self.d_lr
        else:
            return 0.01 * self.d_lr

    def D_train_step(self, inputs):
        """
        Args:
          inputs: one batch input of batch_size images

        Returns:
          loss: the loss of reconstruction the images with the current variables
        """
        # TODO: potentially create a new Z every time this runs??
        # that seems wasteful tho

        # First, normalize all the kernels
        for filter_block in self.filter_blocks:
            D = filter_block.filters
            D.assign(tf.math.l2_normalize(D, axis=(0, 1)))

        # Then, optimize Z for the current D
        for _ in range(self.z_epochs):
            self.Z_train_step(inputs)

        # Then update D based on the loss gradient
        with tf.GradientTape() as tape:
            loss = self.zblock(inputs)

        D_variables = list(itertools.chain.from_iterable(
            filter_block.trainable_variables for filter_block in self.filter_blocks))
        gradients = tape.gradient(loss, D_variables)
        self.d_opt.apply_gradients(zip(gradients, D_variables))

        self.loss_metric(loss)
        return loss

    def Z_train_step(self, inputs):
        # Then, optimize Z for the current D
        # TODO: use the same algorithm the authors did in the paper instead of
        # hacking a standard optimizer on top of it
        with tf.GradientTape() as tape:
            loss = self.zblock(inputs)

        gradients = tape.gradient(loss, self.zblock.trainable_variables)
        self.z_opt.apply_gradients(zip(gradients, self.zblock.trainable_variables))

    # Yoiked from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/densenet/distributed_train.py
    def custom_loop(self, train_dist_dataset, strategy):
        """Custom training loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          strategy: Distribution strategy.
        Returns:
          train_loss
        """

        def distributed_train_epoch(ds):
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in ds:
                per_replica_loss = strategy.run(self.D_train_step, args=(one_batch,))
                total_loss += strategy.reduce(
                        tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                num_train_batches += 1
            return total_loss, num_train_batches

        distributed_train_epoch = tf.function(distributed_train_epoch)

        for epoch in range(self.epochs):
            self.d_opt.learning_rate = self.decay(epoch)

            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_dataset)

            template = 'Epoch: {}, Train Loss: {}'

            print(
              template.format(epoch,
                              train_total_loss / num_train_batches))

            for i in range(len(self.filter_blocks)):
                filter_block = self.filter_blocks[i]
                filter_block.save_weights(f"checkpoints/filter{i}_epoch{epoch}.h5")


        return train_total_loss / num_train_batches


def main(epochs,
         z_epochs=10,
         num_gpu=1,
         batch_size=64,
         buffer_size=5):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    devices = ['/device:GPU:{}'.format(i) for i in range(num_gpu)]
    strategy = tf.distribute.MirroredStrategy(devices)

    import garage_dataset

    train_dataset = garage_dataset.create_dataset(
        buffer_size, batch_size, 'channels_last')

    with strategy.scope():
        trainer = Train(epochs, z_epochs, batch_size,
                        height=224,
                        width=224,
                        num_channels=3,
                        scales=[(13, 128), (27, 64)],
                        d_lr=0.1,
                        z_lr=0.1,
                        alpha=0.01)

        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        trainer.custom_loop(train_dist_dataset, strategy)

if __name__ == "__main__":
    main(epochs=100, batch_size=16)
