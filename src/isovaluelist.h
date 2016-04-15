#include <iostream>
#include <vector>

using namespace std;

template<class G>
class CIsoValueList
{

public:

	void load_values()
	{
		float f;
		cout << "Enter Isovalues: ";
		while (cin >> f)
			m_isovalues.push_back(f);
			
		m_size = m_isovalues.size();

		//for (size_t i = 0; i < m_isovalues.size(); ++i) cout << m_isovalues[i] << ' ';
		//cout << endl;

	}

	int& size()				{	return m_size;			  }
	float& operator[](int i)	{	return m_isovalues[i];	  }

private:
	vector<float> m_isovalues;
	int m_size;
};