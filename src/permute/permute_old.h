
#include <iostream>
#include <numeric>
#include <vector>
#include <memory.h>
#include <memory>

namespace nt{
namespace permute{
class PermIndexItND{
	protected:
		std::vector<int64_t> increment_nums;
		std::vector<int64_t>::const_iterator adds;
		int64_t start;
		int64_t current_index;
	public:
		explicit PermIndexItND(std::vector<int64_t>, std::vector<int64_t>::const_iterator, const int64_t, const int64_t);
		virtual PermIndexItND& operator++();
		PermIndexItND operator++(int);
		virtual PermIndexItND& operator+=(int64_t ad);
		PermIndexItND operator+(int64_t ad);
		friend bool operator==(const PermIndexItND&, const PermIndexItND&);
		friend bool operator!=(const PermIndexItND&, const PermIndexItND&);
		int64_t& operator*(){return current_index;}

};

class PermIndexItND_contig : public PermIndexItND{
	public:
		explicit PermIndexItND_contig(std::vector<int64_t>, std::vector<int64_t>::const_iterator, const int64_t, const int64_t); 
		PermIndexItND_contig& operator++() override;
		PermIndexItND_contig& operator+=(int64_t ad) override;
		friend bool operator==(const PermIndexItND_contig&, const PermIndexItND_contig&);
		friend bool operator!=(const PermIndexItND_contig&, const PermIndexItND_contig&);
};

class PermIndexIt2D : public PermIndexItND{
	public:
		explicit PermIndexIt2D(std::vector<int64_t>, std::vector<int64_t>::const_iterator, const int64_t, const int64_t);
		PermIndexIt2D& operator++() override;
		PermIndexIt2D& operator+=(int64_t ad) override;
		friend bool operator==(const PermIndexIt2D&, const PermIndexIt2D&);
		friend bool operator!=(const PermIndexIt2D&, const PermIndexIt2D&);
};

class PermIndexIt3D : public PermIndexItND{
	public:
		explicit PermIndexIt3D(std::vector<int64_t>, std::vector<int64_t>::const_iterator, const int64_t, const int64_t);
		PermIndexIt3D& operator++() override;
		PermIndexIt3D& operator+=(int64_t ad) override;
		friend bool operator==(const PermIndexIt3D&, const PermIndexIt3D&);
		friend bool operator!=(const PermIndexIt3D&, const PermIndexIt3D&);
};

class PermIndexIt4D : public PermIndexItND{
	public:
		explicit PermIndexIt4D(std::vector<int64_t>, std::vector<int64_t>::const_iterator, const int64_t, const int64_t);
		PermIndexIt4D& operator++() override;
		PermIndexIt4D& operator+=(int64_t ad) override;
		friend bool operator==(const PermIndexIt4D&, const PermIndexIt4D&);
		friend bool operator!=(const PermIndexIt4D&, const PermIndexIt4D&);
};

class PermIndexIt5D : public PermIndexItND{
	public:
		explicit PermIndexIt5D(std::vector<int64_t>, std::vector<int64_t>::const_iterator, const int64_t, const int64_t);
		PermIndexIt5D& operator++() override;
		PermIndexIt5D& operator+=(int64_t ad) override;
		friend bool operator==(const PermIndexIt5D&, const PermIndexIt5D&);
		friend bool operator!=(const PermIndexIt5D&, const PermIndexIt5D&);
};

class PermND{
	protected:
		const std::vector<int64_t> &_strides, &_shape;
		int64_t start;
	public:
		PermND(const std::vector<int64_t>&, const std::vector<int64_t>&);
		bool is_contiguous() const;
		virtual int64_t get_index(const int64_t i) const;

		virtual std::shared_ptr<PermIndexItND> begin(int64_t i =0) const;
		std::shared_ptr<PermIndexItND> end() const;
		virtual std::vector<int64_t> return_indexes() const;
		virtual std::vector<int64_t> return_indexes(std::vector<int64_t>::const_iterator begin_a) const;
		virtual std::vector<int64_t> return_indexes(const int64_t* begin_a) const;
		virtual void perm_in_place(int64_t* first, int64_t* last, const int64_t& total) const;
		virtual void perm_in_place(void** original, void** vals) const;
};


class Perm2D : public PermND{
	public:
		Perm2D(const std::vector<int64_t>&, const std::vector<int64_t>&);
		int64_t get_index(const int64_t i) const override;
		std::shared_ptr<PermIndexItND> begin(int64_t i =0) const override;
		std::vector<int64_t> return_indexes() const override;
		std::vector<int64_t> return_indexes(std::vector<int64_t>::const_iterator begin_a) const override;
		std::vector<int64_t> return_indexes(const int64_t* begin_a) const override;
		void perm_in_place(int64_t* first, int64_t* last, const int64_t& total) const override;
};

class Perm3D : public PermND{
	public:
		Perm3D(const std::vector<int64_t>&, const std::vector<int64_t>&);
		int64_t get_index(const int64_t i) const override;
		std::shared_ptr<PermIndexItND> begin(int64_t i =0) const override;
		std::vector<int64_t> return_indexes() const override;
		std::vector<int64_t> return_indexes(std::vector<int64_t>::const_iterator begin_a) const override;
		std::vector<int64_t> return_indexes(const int64_t* begin_a) const override;
};

class Perm4D : public PermND{
	public:
		Perm4D(const std::vector<int64_t>&, const std::vector<int64_t>&);
		int64_t get_index(const int64_t i) const override;
		std::shared_ptr<PermIndexItND> begin(int64_t i =0) const override;
		std::vector<int64_t> return_indexes() const override;
		std::vector<int64_t> return_indexes(std::vector<int64_t>::const_iterator begin_a) const override;
		std::vector<int64_t> return_indexes(const int64_t* begin_a) const override;
};

class Perm5D : public PermND{
	public:
		Perm5D(const std::vector<int64_t>&, const std::vector<int64_t>&);
		int64_t get_index(const int64_t i) const override;
		std::shared_ptr<PermIndexItND> begin(int64_t i =0) const override;
		std::vector<int64_t> return_indexes() const override;
		std::vector<int64_t> return_indexes(std::vector<int64_t>::const_iterator begin_a) const override;
		std::vector<int64_t> return_indexes(const int64_t* begin_a) const override;
};

std::unique_ptr<PermND> create_perm(const std::vector<int64_t>&, const std::vector<int64_t>&);

}
}
