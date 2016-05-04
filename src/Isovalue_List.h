#include <iostream>
#include <vector>

template<typename T>
class CIsovalueList : public std::vector<T>
{
public:
	
	inline CIsovalueList<T>* operator->() { return this; }

	void load_values()
	{
		float f;
		std::cout << "Enter Isovalues: ";
		while (std::cin >> f)
			this->push_back(f);
	}
};