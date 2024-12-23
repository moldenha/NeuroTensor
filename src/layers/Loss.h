#ifndef _NT_LAYER_LOSS_H_
#define _NT_LAYER_LOSS_H_

#include "TensorGrad.h"
#include "Layer.h"
#include "layers.h"

namespace nt{
class Loss{
	Layer& l; //holds reference of the layer
	protected:
		Tensor dx;
	public:
		Loss(Layer&);
		virtual void operator()(const TensorGrad&, const Tensor&);
		void backward_no_update(); //default route also updates
		void backward();
		void update();
		Scalar item() const;
};


class MSELoss : public Loss{
	public:
		MSELoss(Layer&);
		void operator()(const TensorGrad&, const Tensor&) override;
};

}


#endif //_NT_LAYER_LOSS_H_
