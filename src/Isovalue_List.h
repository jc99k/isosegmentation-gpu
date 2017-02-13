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
		std::cout << "Enter Isovalues: \n";
		while (std::cin >> f){
			if (f < 0) break;
			this->push_back(f);
		}
	}
};