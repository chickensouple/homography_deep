from spatial_transformer import spatial_transformer_network
import tensorflow as tf
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from vgg import VGG19

def test_spatial_transformer():
	# image = misc.imread("cat.jpeg") 
	image = misc.imread("dog.jpg") 
	image = np.array(image, dtype=np.float32) / 255
	image = np.sum(image, axis=2, keepdims=True) / 3

	image = np.array([image])


	# plt.imshow(image[0, :, :, 0])
	# plt.show()
	
	angle = np.pi / 4
	theta = np.array([np.cos(angle), -np.sin(angle), 0.1, np.sin(angle), np.cos(angle), 0])
	theta = np.array([theta])

	image_input = tf.placeholder(tf.float32, [None, 224, 224, 1])
	theta_input = tf.placeholder(tf.float32, [None, 6])

	image_out = spatial_transformer_network(image_input, theta_input)

	sess = tf.InteractiveSession()
	# tf.initialize_variables()

	fd = {image_input: image, theta_input: theta}
	out_image = sess.run(image_out, feed_dict=fd)
	out_image[out_image < 0] = 0

	image_data = np.concatenate([image, out_image], axis=3)
	# np.save('cat_affine_data', image_data)
	np.save('dog_affine_data', image_data)
	print("shape: " + str(image_data.shape))


	# plt.imshow(out_image[0, :, :, 0])
	# plt.show()

def test():
	vgg = VGG19(2)

	# writer = tf.train.SummaryWriter("logs", graph=tf.get_default_graph())

	data = np.load('cat_affine_data.npy')
	# data = np.load('dog_affine_data.npy')

	for i in range(200):
		loss, params, images = vgg.train(data)
		print("Iteration " + str(i) + ": " + str(loss))
		print("Params: " + str(params))




	plt.figure(1)
	plt.imshow(data[0, :, :, 0])
	plt.title('image1')

	plt.figure(2)
	plt.imshow(data[0, :, :, 1])
	plt.title('image2')

	plt.figure(3)
	plt.imshow(images[0, :, :, 0])
	plt.title("transformed")


	plt.show()



if __name__ == '__main__':
	# test_spatial_transformer()
	test()
