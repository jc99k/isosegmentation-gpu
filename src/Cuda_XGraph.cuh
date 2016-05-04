#ifndef CUDA_XGRAPH_H
#define CUDA_XGRAPH_H

template<class X>
class CCudaXGraph
{
public:
	typedef X XSpace;   /////////////

	inline XSpace* operator->()	    { return &m_XSpace; }
	void Init();

private:
	XSpace m_XSpace;

};

#endif //CUDA_XGRAPH_H