#include <_types/_uint32_t.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <memory.h>

namespace nt{
namespace permute{
class PermIndexItND{
	protected:
		std::vector<uint32_t> increment_nums;
		std::vector<uint32_t>::const_iterator adds;
		std::size_t start;
		std::size_t current_index;
	public:
		explicit PermIndexItND(std::vector<uint32_t>, std::vector<uint32_t>::const_iterator, const std::size_t, const std::size_t);
		virtual PermIndexItND& operator++();
		PermIndexItND operator++(int);
		virtual PermIndexItND& operator+=(uint32_t ad);
		PermIndexItND operator+(uint32_t ad);
		friend bool operator==(const PermIndexItND&, const PermIndexItND&);
		friend bool operator!=(const PermIndexItND&, const PermIndexItND&);
		std::size_t& operator*(){return current_index;}

};

class PermIndexItND_contig : public PermIndexItND{
	public:
		explicit PermIndexItND_contig(std::vector<uint32_t>, std::vector<uint32_t>::const_iterator, const std::size_t, const std::size_t); 
		PermIndexItND_contig& operator++() override;
		PermIndexItND_contig& operator+=(uint32_t ad) override;
		friend bool operator==(const PermIndexItND_contig&, const PermIndexItND_contig&);
		friend bool operator!=(const PermIndexItND_contig&, const PermIndexItND_contig&);
};

class PermIndexIt2D : public PermIndexItND{
	public:
		explicit PermIndexIt2D(std::vector<uint32_t>, std::vector<uint32_t>::const_iterator, const std::size_t, const std::size_t);
		PermIndexIt2D& operator++() override;
		PermIndexIt2D& operator+=(uint32_t ad) override;
		friend bool operator==(const PermIndexIt2D&, const PermIndexIt2D&);
		friend bool operator!=(const PermIndexIt2D&, const PermIndexIt2D&);
};

class PermIndexIt3D : public PermIndexItND{
	public:
		explicit PermIndexIt3D(std::vector<uint32_t>, std::vector<uint32_t>::const_iterator, const std::size_t, const std::size_t);
		PermIndexIt3D& operator++() override;
		PermIndexIt3D& operator+=(uint32_t ad) override;
		friend bool operator==(const PermIndexIt3D&, const PermIndexIt3D&);
		friend bool operator!=(const PermIndexIt3D&, const PermIndexIt3D&);
};

class PermIndexIt4D : public PermIndexItND{
	public:
		explicit PermIndexIt4D(std::vector<uint32_t>, std::vector<uint32_t>::const_iterator, const std::size_t, const std::size_t);
		PermIndexIt4D& operator++() override;
		PermIndexIt4D& operator+=(uint32_t ad) override;
		friend bool operator==(const PermIndexIt4D&, const PermIndexIt4D&);
		friend bool operator!=(const PermIndexIt4D&, const PermIndexIt4D&);
};

class PermIndexIt5D : public PermIndexItND{
	public:
		explicit PermIndexIt5D(std::vector<uint32_t>, std::vector<uint32_t>::const_iterator, const std::size_t, const std::size_t);
		PermIndexIt5D& operator++() override;
		PermIndexIt5D& operator+=(uint32_t ad) override;
		friend bool operator==(const PermIndexIt5D&, const PermIndexIt5D&);
		friend bool operator!=(const PermIndexIt5D&, const PermIndexIt5D&);
};

class PermND{
	protected:
		const std::vector<uint32_t> &_strides, &_shape;
		std::size_t start;
	public:
		PermND(const std::vector<uint32_t>&, const std::vector<uint32_t>&);
		bool is_contiguous() const;
		virtual std::size_t get_index(const std::size_t i) const;

		virtual std::shared_ptr<PermIndexItND> begin(std::size_t i =0) const;
		std::shared_ptr<PermIndexItND> end() const;
		virtual std::vector<std::size_t> return_indexes() const;
		virtual std::vector<std::size_t> return_indexes(std::vector<std::size_t>::const_iterator begin_a) const;
		virtual std::vector<std::size_t> return_indexes(const std::size_t* begin_a) const;
		virtual void perm_in_place(std::size_t* first, std::size_t* last, const std::size_t& total) const;
		virtual void perm_in_place(void** original, void** vals) const;
};


class Perm2D : public PermND{
	public:
		Perm2D(const std::vector<uint32_t>&, const std::vector<uint32_t>&);
		std::size_t get_index(const std::size_t i) const override;
		std::shared_ptr<PermIndexItND> begin(std::size_t i =0) const override;
		std::vector<std::size_t> return_indexes() const override;
		std::vector<std::size_t> return_indexes(std::vector<std::size_t>::const_iterator begin_a) const override;
		std::vector<std::size_t> return_indexes(const std::size_t* begin_a) const override;
		void perm_in_place(std::size_t* first, std::size_t* last, const std::size_t& total) const override;
};

class Perm3D : public PermND{
	public:
		Perm3D(const std::vector<uint32_t>&, const std::vector<uint32_t>&);
		std::size_t get_index(const std::size_t i) const override;
		std::shared_ptr<PermIndexItND> begin(std::size_t i =0) const override;
		std::vector<std::size_t> return_indexes() const override;
		std::vector<std::size_t> return_indexes(std::vector<std::size_t>::const_iterator begin_a) const override;
		std::vector<std::size_t> return_indexes(const std::size_t* begin_a) const override;
};

class Perm4D : public PermND{
	public:
		Perm4D(const std::vector<uint32_t>&, const std::vector<uint32_t>&);
		std::size_t get_index(const std::size_t i) const override;
		std::shared_ptr<PermIndexItND> begin(std::size_t i =0) const override;
		std::vector<std::size_t> return_indexes() const override;
		std::vector<std::size_t> return_indexes(std::vector<std::size_t>::const_iterator begin_a) const override;
		std::vector<std::size_t> return_indexes(const std::size_t* begin_a) const override;
};

class Perm5D : public PermND{
	public:
		Perm5D(const std::vector<uint32_t>&, const std::vector<uint32_t>&);
		std::size_t get_index(const std::size_t i) const override;
		std::shared_ptr<PermIndexItND> begin(std::size_t i =0) const override;
		std::vector<std::size_t> return_indexes() const override;
		std::vector<std::size_t> return_indexes(std::vector<std::size_t>::const_iterator begin_a) const override;
		std::vector<std::size_t> return_indexes(const std::size_t* begin_a) const override;
};

std::unique_ptr<PermND> create_perm(const std::vector<uint32_t>&, const std::vector<uint32_t>&);

}
}
