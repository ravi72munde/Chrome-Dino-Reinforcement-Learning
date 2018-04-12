
import pickle
import cv2

from PIL import ImageGrab
from PIL import Image


class utils:
	#processing image as required
	def process_img(image):
	    #game is already in grey scale canvas, canny to get only edges and reduce unwanted objects(clouds)
	#     image = cv2.Canny(image, threshold1 = 100, threshold2 = 200)
	#     image = image[10:140,0:200] #img[y:y+h, x:x+w]
	#     image = resized_image = cv2.resize(image, (80, 80)) 
	    image = cv2.resize(image, (0,0), fx = 0.15, fy = 0.10)
	    image = image[2:38,10:50] #img[y:y+h, x:x+w]
	    image = cv2.Canny(image, threshold1 = 200, threshold2 = 200)
	    
	    return  image


	def grab_screen(_driver = None):
	    screen =  np.array(ImageGrab.grab(bbox=(0,180,400,400)))
	    image = process_img(screen)
	    if _driver!=None:
		image = _driver.get_screenshot_as_png()
	    return image


	def save_obj(obj, name ):
	    with open('objects/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def load_obj(name ):
	    with open('objects/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

	def show_img(graphs = False):
	    """
	    Show images in new window
	    """
	    while True:
		screen = (yield)
		window_title = "logs" if graphs else "game_play"
		cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
		imS = cv2.resize(screen, (800, 400)) 
		cv2.imshow(window_title, screen)
	#         cv2.imwrite("screenshot"+str(frame)+".png",screen)
		if (cv2.waitKey(1) & 0xFF == ord('q')):
		    cv2.destroyAllWindows()
		    break

	def show_plots(realtime = True,t=0):
	    fig, axs = plt.subplots(ncols=2,nrows =2)
	    loss_df['loss'] = loss_df['loss'].astype('float') 
	    loss_df.plot(use_index=True,ax=axs[0,0])
	    scores_df.plot(ax=axs[0,1])
	    sns.distplot(actions_df,ax=axs[1,0])
	#     q_max_df.plot(ax = axs[1,1])
	    imgg = fig.canvas.draw()
	    graph_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	    cv2.imwrite("logs/progress/pg"+str(t)+".png",graph_img) if realtime else 0


