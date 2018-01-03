#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//CanonicalAxisIndex是Blob的成员函数，设置shape_的轴，并返回该轴
//layer_param_是Layer中定义的LayerParameter类型的变量，softmax_param是SoftmaxParameter类型，沿着axis执行SoftmaxLayer，axis一般取默认值0

  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
//bottom[0].shape(softmax_axis_)表示第一维的大小，mult_dims此时是指1个值为bottom[0]->shape(softmax_axis_)的向量
//bottom的维度是N*channels*W*H，mult_dims就表示包含1个值为channels的容器

  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);//用Dtype(1)初始化
  outer_num_ = bottom[0]->count(0, softmax_axis_);//outer_num_表示N，N是指每次输入的样本数，每个bottom有N*Channel*W*H，N指的是batch_size。
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);//inner_num_表示W*H，count返回从第2到最后一维的维度乘积，也就是W*H
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;//scale_一个中间Blob，用于hold一些临时结果，这里将它第一维大小设为1
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_); //channels表示最终分类的个数，softmax_axis_的值为1，bottom[0]维度是N*Channel*W*H，
  //channels在softmax层不再表示通道，因为上一层的全连接层已经有了channels个输出，channels表示分类的个数。

  int dim = bottom[0]->count() / outer_num_;//dim是指bottom[0]元素个数，Channels*W*H
  caffe_copy(bottom[0]->count(), bottom_data, top_data);//把bottom_data赋给top_data
//对于mnist数据库来讲，channels为10，下面的for循环表示，对每个像素点，取10个里面的较大值，放到scale_data对应位置

  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);/////-------------
      }
    }
    // subtraction
            //求矩阵的差放入top_data中，公式：top_data = -1*sum_multiplier_*scale_data + top_data
        //sum_multiplier_是channels*1的矩阵，每个元素值为1
        //scale_data是1*(N*W)的矩阵
        //top_data是channels*(N*W)的矩阵
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);//CblasNoTrans：指不做转置
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);//对top_data的每个像素点做幂运算c
    // sum after exp
     //对top_data转置，每一列都加到一起，也就是对应像素点的channels个值相加，放到scale_data中
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
     //求在每一个分类里面的概率值
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);//对每个channel除以scale_data，存到top_data里面
      top_data += inner_num_;//这里的+=inner_num_是指top_data的偏移量。
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
