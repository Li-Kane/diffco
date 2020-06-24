import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.svm import SVC
from scipy import ndimage
from tqdm import tqdm
from time import time

class Obstacle:
    def __init__(self, kind, position, size, cost=np.inf):
        self.kind = kind
        if self.kind not in ['circle', 'rect']:
            raise NotImplementedError('Obstacle kind {} not supported'.format(kind))
        self.position = np.array(position)
        self.size = np.array(size) if kind != 'rect' or (isinstance(size, (list, tuple, np.ndarray)) and len(size) == len(position)) else np.array([size, size])
        self.cost = cost
    
    def is_collision(self, point):
        if self.kind == 'circle':
            return np.linalg.norm(self.position-point) < self.size/2
        elif self.kind == 'rect':
            return np.all(np.abs(self.position-point) < self.size/2)
        else:
            raise NotImplementedError('Obstacle kind {} not supported'.format(self.kind))
    
    def get_cost(self):
        return self.cost

class CollisionChecker():
    def __init__(self, obstacles):
        self.obstacles = obstacles
    
    def is_collision(self, point):
        return any(map(lambda obs: obs.is_collision(point), self.obstacles))
    
    def line_collision(self, start, target, res=50):
        points = map(lambda i: start + (target - start)/res*i, range(res))
        return any(map(lambda p: self.is_collision(p), points))
    
    def predict(self, X):
        return np.array(list(map(lambda x: self.is_collision(x), X)))


class Fastron(CollisionChecker):
    def __init__(self, obstacles, kernel_func='rq', gamma=1, beta=1, gt_checker=None):
        super().__init__(obstacles)
        self.gt_checker = gt_checker if gt_checker is not None else CollisionChecker(self.obstacles)
        self.train_method = None
        self.kernel_func = kernel.RQKernel(gamma) if kernel_func=='rq' else kernel_func
        # self.gamma = self.kernel_func.gamma #C0.2 # 1/(2*self.support_points.var())
        self.beta = beta
    
    def initialize(self, num_init_points=100):
        self.support_points = np.random.random((num_init_points, 2)) * 10
        self.gains = np.zeros(num_init_points)
        # K = np.tile(self.support_points[np.newaxis, :], (num_init_points, 1, 1))
        # self.kernel_matrix = (self.support_points@self.support_points.T+1)**2
        # self.kernel_matrix = 1/(1+self.gamma/2*np.sum((K-K.transpose(1, 0, 2))**2, axis=2))**2
        self.kernel_matrix = np.zeros((num_init_points, num_init_points))
        self.hypothesis = np.zeros(num_init_points)
        self.max_n_support = 200

    def train(self, max_iteration=1000, method='original'):
        self.train_method = method
        time_start = time()
        if method == 'original':
            self.train_original(max_iteration)
        elif method == 'sgd':
            self.train_sgd(max_iteration)
        elif method == 'svm':
            self.train_svm()
        
        self.support_points = self.support_points[self.gains != 0]
        self.hypothesis = self.hypothesis[self.gains != 0]
        self.y = self.y[self.gains != 0]
        self.gains = self.gains[self.gains != 0]
        time_elapsed = time() - time_start
        print('{} training done. {:.4f} secs cost'.format(method, time_elapsed))
        
        

    def train_original(self, max_iteration=1000):
        self.y = np.zeros(len(self.support_points))
        for i in range(len(self.support_points)):
            self.y[i] = 1 if self.gt_checker.is_collision(self.support_points[i]) else -1
        
        print('Fastron training...')
        for it in tqdm(range(max_iteration)):
            margin = self.y * self.hypothesis
            min_i = np.argmin(margin)  #1./
            if self.kernel_matrix[min_i, min_i] == 0:
                self.kernel_matrix[min_i] = self.kernel_func(self.support_points[min_i], self.support_points)
                self.kernel_matrix[:, min_i] = self.kernel_matrix[min_i]
            if margin[min_i] <= 0:
                delta_gain = (self.beta**((1+self.y[min_i])/2)*self.y[min_i] - self.hypothesis[min_i])/self.kernel_matrix[min_i, min_i]# 
                # assert delta_gain > -1000 and delta_gain < 1000
                self.gains[min_i] += delta_gain
                assert delta_gain < 1000 and delta_gain > -1000 and self.kernel_matrix[min_i].max() < 1000
                self.hypothesis += delta_gain * self.kernel_matrix[min_i]
                # self.hypothesis[min_margin_idx] = self.gains @ self.kernel_matrix[:, min_margin_idx]
                continue
            
            modified_margin = self.y*(self.hypothesis - self.gains * np.diag(self.kernel_matrix)) * (self.gains != 0 )  # 
            max_margin_idx = np.argmax(modified_margin)
            if modified_margin[max_margin_idx] > 0 and np.sum(self.gains != 0) > 1:
                self.hypothesis -= self.gains[max_margin_idx]*self.kernel_matrix[max_margin_idx]
                self.gains[max_margin_idx] = 0
                continue

            break

        print('Ended at iteration {}'.format(it))

        
        print('ACC: {}'.format(np.sum((self.hypothesis > 0) == (self.y > 0)) / len(self.y)))
        
    def train_sgd(self, max_iteration=1000):
        self.y = np.zeros(len(self.support_points))
        for i in range(len(self.support_points)):
            self.y[i] = 1 if self.gt_checker.is_collision(self.support_points[i]) else -1
        y = torch.tensor(self.y)
        K = torch.tensor(self.kernel_matrix)
        gains = torch.tensor(self.gains, requires_grad=True)

        # self.grad = self.kernel_matrix@self.y
        for it in range(max_iteration):
            margin = (gains@K)*y
            margin[margin > 0] = torch.log(1 + margin[margin > 0])
            sum_margin = margin.sum()
            gains.grad = None
            sum_margin.backward()
            gains.data += 0.001 * gains.grad
            # self.gains /= np.linalg.norm(self.gains)
        
        self.gains = gains.detach().numpy()
    
    def train_greedy_sgd(self, max_iteration=1000):
        # TODO: Code an update rule that increases the smallest margin and 
        # normalize the largest margin to be 1
        self.y = np.zeros(len(self.support_points))
        for i in range(len(self.support_points)):
            self.y[i] = 1 if self.gt_checker.is_collision(self.support_points[i]) else -1
        y = torch.tensor(self.y)
        K = torch.tensor(self.kernel_matrix)
        gains = torch.tensor(self.gains, requires_grad=True)

        # self.grad = self.kernel_matrix@self.y
        for it in range(max_iteration):
            margin = (gains@K)*y
            margin[margin > 0] = torch.log(1 + margin[margin > 0])
            sum_margin = margin.sum()
            gains.grad = None
            sum_margin.backward()
            gains.data += 0.001 * gains.grad
            # self.gains /= np.linalg.norm(self.gains)
        
        self.gains = gains.detach().numpy()
    
    def train_svm(self, max_iteration=1000):
        self.y = np.zeros(len(self.support_points))
        for i in range(len(self.support_points)):
            self.y[i] = 1 if self.gt_checker.is_collision(self.support_points[i]) else -1
        self.svm = SVC(C=1e8, kernel='rbf', max_iter=max_iteration) #/4self.kernel_func, gamma=1/(2*self.support_points.var())
        self.svm.fit(self.support_points, self.y)
        # self.support_points = self.svm.support_vectors_
        self.gains[self.svm.support_] = self.svm.dual_coef_.reshape(-1)
        self.intercept = self.svm.intercept_
        self.svm_gamma = self.svm._gamma#self.svm.gamma if isinstance(self.svm.gamma, (float, int)) else 1/(self.support_points.shape[1]*self.support_points.var())
        print('SVM Gamma: {}'.format(self.svm.gamma))
        # print('Intercept:', self.intercept)
        # print('Gains: ', self.svm.dual_coef_)
        print('ACC: {}'.format(np.sum((self.svm.predict(self.support_points) > 0) == (self.y > 0)) / len(self.y)))
        
    def is_collision(self, point):
        return self.score(point) > 0
    
    def score(self, point):
        if self.train_method == 'original':
            return self.score_original(point)
        elif self.train_method == 'sgd':
            self.score_original(point)
        elif self.train_method == 'svm':
            self.score_svm(point)

    def score_original(self, point):
        # kernel_values = 1/(1+self.gamma/2*np.sum((self.support_points-point)**2, axis=1))**2
        kernel_values = self.kernel_func(point, self.support_points)
        score = self.gains@kernel_values
        return score
    
    def score_nn(self, point):
        dif_abs = np.abs((self.support_points-point))
        dist = np.sqrt(np.sum(dif_abs**2, axis=1))
        dist -= dist.min()
        # print(dist.min())
        kernel_values = self.kernel_func(point, self.support_points)
        # nn_idx = np.argmax(kernel_values)
        # score = self.hypothesis[nn_idx] * (2-kernel_values[nn_idx])
        score = self.gains@kernel_values
        return score
    
    def score_svm(self, point):
        # kernel_values = self.kernel_func(point, self.support_points)
        kernel_values = np.exp(-self.svm_gamma*np.sum((self.support_points-point)**2, axis=1))
        return (self.gains)@kernel_values + self.intercept
        # return self.svm.decision_function(point.reshape(1, -1))
    
def vis(model, size=100, seed=2019):
    if isinstance(size, int):
        size = [size, size]
    xx, yy = np.meshgrid(np.linspace(0, 10, size[0]), np.linspace(0, 10, size[1]))
    grid_points = np.stack([xx, yy], axis=2).reshape((-1, 2))
    fig, ax = plt.subplots(figsize=(28, 10)) #(figsize=(42, 10)) (14, 10) 

    np.random.seed(seed)
    model.initialize(3000)
    model.train(200000, method='original')
    real_support_points = model.support_points
    grid_score = np.fromiter(map(model.score_original, grid_points), np.float).reshape((size[0], size[1]))
    # ax1 = plt.subplot(1,2,1)
    ax1 = plt.subplot(1,2,1)
    c = ax1.pcolormesh(xx, yy, grid_score, cmap='RdBu_r', vmin=-np.abs(grid_score).max(), vmax=np.abs(grid_score).max())
    ax1.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
    ax1.contour(xx, yy, (grid_score).astype(float), levels=0)
    ax1.axis('equal')
    fig.colorbar(c, ax=ax1)
    sparse_score = grid_score[::10, ::10]
    score_grad_x = -ndimage.sobel(sparse_score, axis=1)
    score_grad_y = -ndimage.sobel(sparse_score, axis=0)
    score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    ax1.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=30, color='red')
    ax1.set_title('Original Fastron (kernel={}), no. of support points = {}'.format(model.kernel_func.__class__.__name__+str(model.kernel_func.__dict__), len(real_support_points)))

    # grid_nn_score = np.fromiter(map(model.score_nn, grid_points), np.float).reshape((size[0], size[1]))
    # ax2 = plt.subplot(1,3,2)
    # c = ax2.pcolormesh(xx, yy, grid_nn_score, cmap='RdBu_r', vmin=-np.abs(grid_nn_score).max(), vmax=np.abs(grid_nn_score).max())
    # ax2.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
    # ax2.contour(xx, yy, (grid_nn_score).astype(float), levels=0)
    # ax2.axis('equal')
    # fig.colorbar(c, ax=ax2)
    # sparse_score = grid_nn_score[::10, ::10]
    # score_grad_x = -ndimage.sobel(sparse_score, axis=1)
    # score_grad_y = -ndimage.sobel(sparse_score, axis=0)
    # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    # ax2.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=30, color='red')
    # ax2.set_title('NN inference, no. of support points = {}'.format(len(real_support_points)))

    np.random.seed(seed)
    model.initialize(3000)
    model.train(1000, method='svm')
    real_support_points = model.support_points
    grid_svm_score = np.fromiter(map(model.score_svm, grid_points), np.float).reshape((size[0], size[1]))
    ax3 = plt.subplot(122)
    c = ax3.pcolormesh(xx, yy, grid_svm_score, cmap='RdBu_r', vmin=-np.abs(grid_svm_score).max(), vmax=np.abs(grid_svm_score).max())
    ax3.scatter(real_support_points[:, 0], real_support_points[:, 1], marker='.', c='black')
    ax3.contour(xx, yy, (grid_svm_score).astype(float), levels=0)
    ax3.axis('equal')
    fig.colorbar(c, ax=ax3)
    sparse_score = grid_svm_score[::10, ::10]
    score_grad_x = -ndimage.sobel(sparse_score, axis=1)
    score_grad_y = -ndimage.sobel(sparse_score, axis=0)
    score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    ax3.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, scale=30, color='red')
    ax3.set_title('SVM, no. of support points={}'.format(len(model.support_points)))

    for obs in model.obstacles:
        if obs.kind == 'circle':
            circle_artist = plt.Circle(obs.position, radius=obs.size/2, color=[0, 0, 0, 0.3])
            ax1.add_artist(circle_artist)
            circle_artist = plt.Circle(obs.position, radius=obs.size/2, color=[0, 0, 0, 0.3])
            ax3.add_artist(circle_artist)
        elif obs.kind == 'rect':
            rect_artist = plt.Rectangle(obs.position-obs.size/2, obs.size[0], obs.size[1], color=[0, 0, 0, 0.3])
            ax1.add_artist(rect_artist)
            rect_artist = plt.Rectangle(obs.position-obs.size/2, obs.size[0], obs.size[1], color=[0, 0, 0, 0.3])
            ax3.add_artist(rect_artist)
        else:
            raise NotImplementedError('Unknown obstacle type')

    plt.show()

if __name__ == '__main__':
    import kernel
    obstacles = [
        ('circle', (6, 2), 2),
        # ('circle', (2, 7), 1),
        ('rect', (3.5, 6), (2, 1)),
        ('rect', (4, 7), 1),
        ('rect', (5, 8), (10, 1)),
        ('rect', (7.5, 6), (2, 1)),
        ('rect', (8, 7), 1),]
    obstacles = [Obstacle(*param) for param in obstacles]
    # kernel = kernel.CauchyKernel(100)
    # k = kernel.TangentKernel(0.8, 0)
    k = kernel.RQKernel(0.7)
    # k = kernel.MultiQuadratic(0.7)
    # lambda x, x_prime: -k(x, x_prime)+k(np.array([0, 0])， np.array([[10, 10]]))
    checker = Fastron(obstacles, kernel_func=k, beta=1)
    vis(checker, 200, seed=1917)
 