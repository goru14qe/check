#ifndef Tensor_H
#define Tensor_H

#include "utils/Assert.h"
#include <memory>
#include <array>
#include <type_traits>
#include <iostream>
#include <algorithm>
#include <numeric>

#ifndef NDEBUG
// Remove this define if you want to disable bound checks even in debug builds.
#define OUT_OF_BOUNDS_CHECKS
#endif

#ifdef OUT_OF_BOUNDS_CHECKS
#define OOB_ASSERT(expr) ASSERT(expr)
#else
#define OOB_ASSERT(expr) \
	do {                 \
	} while (false)
#endif

using Index = int32_t;
using Flat_index = int64_t;
// has to be size_t here to allow auto deduction.
template <size_t N>
using Base_index_vec = std::array<Index, N>;

// hash function to enable usage of Index vectors as keys
template <size_t N>
struct std::hash<Base_index_vec<N>> {
	std::size_t operator()(const Base_index_vec<N>& s) const noexcept {
		static_assert(N >= 1);
		// basically boost::hash_combine
		std::size_t seed = s[0];
		for(size_t i = 1; i < N; ++i){
			seed ^= s[1] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

enum struct Storage_order {
	ROW_MAJOR,
	SWITCH_LAST // row-major but for N==4 the last dimension is first
};

// Holds size and implements various index operations but does not hold any data.
template <int N, Storage_order S = Storage_order::ROW_MAJOR>
class Tensor_base {
	static_assert(N > 0, "Tensor dimension must be positive.");
	static_assert(N == 4 || S == Storage_order::ROW_MAJOR, "alternative storage order is only supported for N==4");

public:
	using Index_vec = Base_index_vec<N>;
	static constexpr int Order = N;

	Tensor_base() noexcept
		: m_sizes{}, m_num_elem(0) {}

	explicit Tensor_base(const Index_vec& sizes) noexcept
		: m_sizes(sizes), m_strides(compute_strides(sizes)), m_num_elem(compute_num_elem(sizes)) {}

	// Computes the flat index for the element addressed by idx.
	Flat_index flat_index(const Index_vec& idx) const noexcept;

	// Computes a vector index from a flat index.
	Index_vec index(Flat_index flat) const noexcept;

	const Index_vec& sizes() const noexcept { return m_sizes; }
	const Flat_index num_elem() const noexcept { return m_num_elem; }

	// Check whether the index is inside the bounds of this tensor.
	bool is_valid_idx(const Index_vec& idx) const noexcept;

	static Flat_index compute_num_elem(const Index_vec& sizes) noexcept;
	static Index_vec compute_strides(const Index_vec& sizes) noexcept;
protected:
	void resize(const Index_vec& sizes) noexcept {
		m_sizes = sizes;
		m_strides = compute_strides(sizes);
		m_num_elem = compute_num_elem(sizes);
	}

	Flat_index flat_index(Flat_index base, Index offset) const noexcept;

private:
	Index_vec m_sizes;
	Index_vec m_strides;
	Flat_index m_num_elem;
};

// The main tensor class which owns a contigious block of memory.
// Currently data is stored in row-major order.
template <typename T, int N, Storage_order S = Storage_order::ROW_MAJOR>
class Tensor : public Tensor_base<N, S> {
public:
	using Value_type = T;
	using Index_vec = typename Tensor_base<N>::Index_vec;

	Tensor() noexcept
		: m_data(nullptr) {
	}

	Tensor(const Tensor& oth);
	Tensor(Tensor&& oth) noexcept;

	template <typename Tensor_expr, typename = typename std::enable_if<!std::is_same<Tensor<T, N>, Tensor_expr>::value>::type>
	Tensor(const Tensor_expr& view);

	explicit Tensor(const Index_vec& size)
		: Tensor_base<N,S>(size)
		, m_data(this->num_elem() > 0 ? new T[this->num_elem()] : nullptr) {
	}

	Tensor& operator=(Tensor&& oth) noexcept;
	Tensor& operator=(const Tensor& oth);

	static Tensor zeros(const Index_vec& size);
	static Tensor ones(const Index_vec& size);

	template <typename Tensor_expr, typename = typename std::enable_if<!std::is_same<Tensor<T, N, S>, Tensor_expr>::value>::type>
	Tensor& operator=(const Tensor_expr& view);

	// set every element to zero
	void zero();

	// regular index access
	T& operator[](const Index_vec& idx) {
		OOB_ASSERT(this->is_valid_idx(idx));
		return m_data[this->flat_index(idx)];
	}
	const T& operator[](const Index_vec& idx) const {
		OOB_ASSERT(this->is_valid_idx(idx));
		return m_data[this->flat_index(idx)];
	}

	// fast access through flat index
	T& operator[](const Flat_index& idx) {
		OOB_ASSERT(idx < this->num_elem());
		return m_data[idx];
	}
	const T& operator[](const Flat_index& idx) const {
		OOB_ASSERT(idx < this->num_elem());
		return m_data[idx];
	}

	// lattice based access with cell idx and offset, for 4th-order tensors
	template <typename V = T, typename = typename std::enable_if<N == 4, V>::type>
	T& operator()(Flat_index idx, Index w) {
		return m_data[this->flat_index(idx, w)];
	}
	template <typename V = T, typename = typename std::enable_if<N == 4, V>::type>
	T& operator()(Flat_index idx, Index w) const {
		return m_data[this->flat_index(idx, w)];
	}

	// raw access to the memory
	const T* data() const noexcept { return m_data.get(); }
	T* data() noexcept { return m_data.get(); }

	Tensor& operator*=(T scalar);
	Tensor& operator+=(const Tensor& oth);
	bool operator==(const Tensor& oth) const;

	friend void swap(Tensor<T, N, S>& a, Tensor<T, N, S>& b) noexcept {
		const auto s = a.sizes();
		a.resize(b.sizes());
		b.resize(s);
		std::swap(a.m_data, b.m_data);
	}

private:
	std::unique_ptr<T[]> m_data;
};

// stream operator for easy printing of Index_vec
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& out, const std::array<T, N>& idx) {
	for (const T& v : idx) {
		out << v << " ";
	}
	return out;
}

// common tensor types for alborz
using Vector_field = Tensor<double, 4>;
using Population_field = Tensor<double, 4>;
using Scalar_field = Tensor<double, 3>;
using Solid_field = Tensor<int, 3>;

using Index_vec3 = typename Tensor_base<3>::Index_vec;
using Index_vec4 = typename Tensor_base<4>::Index_vec;

// *************************************************** //
// Tensor views
// Views can be accessed via index like a tensor but do not hold any data.

// Wrapper for the actual tensor to enable value semantics without copying its data.
template <typename T, int N>
class Tensor_ref {
public:
	static constexpr int Order = N;
	using Value_type = T;
	using Index_vec = typename Tensor<T, N>::Index_vec;

	Tensor_ref(Tensor<T, N>& tensor) noexcept
		: m_tensor(&tensor) {}
	Tensor_ref(const Tensor<T, N>& tensor) noexcept
		: m_tensor(&const_cast<Tensor<T, N>&>(tensor)) {}

	operator Tensor<T, N>&() noexcept { return *m_tensor; }
	operator const Tensor<T, N>&() const noexcept { return *m_tensor; }

	const T& operator[](const Index_vec& idx) const { return (*m_tensor)[idx]; }
	T& operator[](const Index_vec& idx) { return (*m_tensor)[idx]; }

	Index_vec index(Flat_index flat) const noexcept { return m_tensor->index(flat); }

	const Index_vec& sizes() const noexcept { return m_tensor->sizes(); }
	Flat_index num_elem() const noexcept { return m_tensor->num_elem(); }

private:
	Tensor<T, N>* m_tensor;
};

namespace details {
template <typename Tensor_expr>
struct Tensor_expr_holder {
	using Type = Tensor_expr;
};
template <typename T, int N>
struct Tensor_expr_holder<Tensor<T, N>> {
	using Type = Tensor_ref<T, N>;
};

// selects the correct wrapper for the tensor expression
template <typename Tensor_expr>
using Tensor_expr_t = typename Tensor_expr_holder<Tensor_expr>::Type;
}  // namespace details

// Reorders the dimensions.
template <typename Tensor_expr>
class Transpose_view : public Tensor_base<Tensor_expr::Order> {
public:
	static constexpr int Order = Tensor_expr::Order;
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = typename Tensor_expr::Index_vec;

	Transpose_view(const Tensor_expr& expr, const Index_vec& dim_order) noexcept
		: Tensor_base<Order>(reorder(expr.sizes(), dim_order))
		, m_expr(expr)
		, m_dim_order(dim_order) {
		ASSERT(is_permutation(dim_order));
	}

	Value_type operator[](const Index_vec& idx) const {
		return m_expr[reorder_inv(idx)];
	}

	Value_type& operator[](const Index_vec& idx) {
		return m_expr[reorder_inv(idx)];
	}

	const Index_vec& dim_order() const noexcept { return m_dim_order; }

private:
	// original order -> transposed
	static Index_vec reorder(const Index_vec& vec, const Index_vec& order) noexcept {
		Index_vec result;
		for (int i = 0; i < Order; ++i) {
			result[i] = vec[order[i]];
		}
		return result;
	}

	// transposed -> original order
	Index_vec reorder_inv(const Index_vec& vec) const noexcept {
		Index_vec result;
		for (int i = 0; i < Order; ++i) {
			result[m_dim_order[i]] = vec[i];
		}
		return result;
	}

	static bool is_permutation(const Index_vec& vec) {
		Index_vec ref;
		std::iota(ref.begin(), ref.end(), 0);
		return std::is_permutation(vec.begin(), vec.end(), ref.begin());
	}

	details::Tensor_expr_t<Tensor_expr> m_expr;
	Index_vec m_dim_order;
};

template <typename Tensor_expr>
Transpose_view<Tensor_expr> transpose(const Tensor_expr& expr,
                                      const typename Tensor_expr::Index_vec& dim_order) noexcept {
	return Transpose_view<Tensor_expr>(expr, dim_order);
}

// short form that completely reverses the order
template <typename Tensor_expr>
Transpose_view<Tensor_expr> transpose(const Tensor_expr& expr) {
	typename Tensor_expr::Index_vec dim_order;
	for (size_t i = 0; i < dim_order.size(); ++i)
		dim_order[i] = Tensor_expr::Order - i - 1;
	return Transpose_view<Tensor_expr>(expr, dim_order);
}

// Permutes the dimensions with an order known at compile time for 0 overhead.
template <typename Tensor_expr, int... Dims>
class Static_transpose_view : public Tensor_base<Tensor_expr::Order> {
public:
	static constexpr int Order = Tensor_expr::Order;
	static_assert(Order == sizeof...(Dims), "Order of the expression needs to match.");
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = typename Tensor_expr::Index_vec;

	Static_transpose_view(const Tensor_expr& expr) noexcept
		: Tensor_base<Order>({expr.sizes()[Dims]...})
		, m_expr(expr) {
	}

	Value_type operator[](const Index_vec& idx) const {
		int i = 0;
		Index_vec idx_res;
		return m_expr[{(idx_res[Dims] = idx[i++])...}];
	}

	Value_type& operator[](const Index_vec& idx) {
		int i = 0;
		Index_vec idx_res;
		return m_expr[{(idx_res[Dims] = idx[i++])...}];
	}

private:
	details::Tensor_expr_t<Tensor_expr> m_expr;
	Index_vec m_dim_order;
};

// View on a sub-tensor with offset and size.
template <typename Tensor_expr>
class Slice_view : public Tensor_base<Tensor_expr::Order> {
public:
	static constexpr int Order = Tensor_expr::Order;
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = typename Tensor_expr::Index_vec;

	Slice_view(const Tensor_expr& expr, const Index_vec& offset, const Index_vec& sizes) noexcept
		: Tensor_base<Order>(sizes), m_expr(expr), m_offset(offset) {
		ASSERT(is_valid());
	}

	Value_type operator[](Index_vec idx) const {
		for (int i = 0; i < Order; ++i) {
			idx[i] += m_offset[i];
		}
		return m_expr[idx];
	}

	Value_type& operator[](Index_vec idx) {
		for (int i = 0; i < Order; ++i) {
			idx[i] += m_offset[i];
		}
		return m_expr[idx];
	}

	const Index_vec& offset() const { return m_offset; }
	const Tensor_expr& base_expr() const { return m_expr; }

	void resize(const Index_vec& offset, const Index_vec& sizes) {
		Tensor_base<Order>::resize(sizes);
		m_offset = offset;
		ASSERT(is_valid());
	}

private:
	bool is_valid() const noexcept {
		for (int i = 0; i < Order; ++i) {
			if (m_offset[i] < 0 || m_offset[i] + this->sizes()[i] > m_expr.sizes()[i])
				return false;
		}
		return true;
	}

	details::Tensor_expr_t<Tensor_expr> m_expr;
	Index_vec m_offset;
};

template <typename Tensor_expr>
Slice_view<Tensor_expr> slice(const Tensor_expr& expr,
                              const typename Tensor_expr::Index_vec& offset,
                              const typename Tensor_expr::Index_vec& sizes) {
	return Slice_view<Tensor_expr>(expr, offset, sizes);
}

// Wrappers for the old pointer based data structure.
template <typename T>
class Ptr3_view : public Tensor_base<3> {
public:
	static constexpr int Order = 3;
	using Value_type = T;
	using Index_vec = typename Tensor_base<3>::Index_vec;

	Ptr3_view(T*** data, const Index_vec& size) noexcept
		: Tensor_base<3>(size), m_data(data) {
		ASSERT(data || this->num_elem() == 0);
	}

	const Value_type& operator[](const Index_vec& idx) const {
		OOB_ASSERT(is_valid_idx(idx));
		return m_data[idx[0]][idx[1]][idx[2]];
	}

	Value_type& operator[](const Index_vec& idx) {
		OOB_ASSERT(is_valid_idx(idx));
		return m_data[idx[0]][idx[1]][idx[2]];
	}

private:
	T*** m_data;
};

template <typename T>
Ptr3_view<T> ptr_view(T*** data, const typename Ptr3_view<T>::Index_vec& size) noexcept {
	return Ptr3_view<T>(data, size);
}

template <typename T>
class Ptr4_view : public Tensor_base<4> {
public:
	static constexpr int Order = 4;
	using Value_type = T;
	using Index_vec = typename Tensor_base<4>::Index_vec;

	Ptr4_view(T**** data, const Index_vec& size) noexcept
		: Tensor_base<4>(size), m_data(data) {
		ASSERT(data || this->num_elem() == 0);
	}

	const Value_type& operator[](const Index_vec& idx) const {
		OOB_ASSERT(is_valid_idx(idx));
		return m_data[idx[0]][idx[1]][idx[2]][idx[3]];
	}

	Value_type& operator[](const Index_vec& idx) {
		OOB_ASSERT(is_valid_idx(idx));
		return m_data[idx[0]][idx[1]][idx[2]][idx[3]];
	}

private:
	T**** m_data;
};

template <typename T>
Ptr4_view<T> ptr_view(T**** data, const typename Ptr4_view<T>::Index_vec& size) noexcept {
	return Ptr4_view<T>(data, size);
}

// A view that applies some function to each element when it is accessed.
template <typename Tensor_expr, typename Filter>
class Filter_view : public Tensor_base<Tensor_expr::Order> {
public:
	static constexpr int Order = Tensor_expr::Order;
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = typename Tensor_expr::Index_vec;

	// @param filter A functor with the signature
	//        Value_type operator()(const Index_vec&, Value_type)
	Filter_view(const Tensor_expr& expr, Filter filter)
		: Tensor_base<Order>(expr.sizes())
		, m_expr(expr)
		, m_filter(filter) {}

	Value_type operator[](const Index_vec& idx) const {
		return m_filter(m_expr[idx], idx);
	}

	// The ability to write through this view does not make much sense here
	// but currently no concept for read only views exists.
	Value_type& operator[](const Index_vec& idx) {
		return m_expr[idx];
	}

private:
	details::Tensor_expr_t<Tensor_expr> m_expr;
	Filter m_filter;
};

template <typename Tensor_expr, typename Filter>
Filter_view<Tensor_expr, Filter> filter_view(const Tensor_expr& expr, Filter filter) {
	return Filter_view<Tensor_expr, Filter>(expr, filter);
}

// A view that applies some function to each element when it is accessed.
/*template <typename Tensor_expr, typename Filter>
class Mask_view : public Tensor_base<Tensor_expr::Order> {
public:
    static constexpr int Order = Tensor_expr::Order;
    using Value_type = typename Tensor_expr::Value_type;
    using Index_vec = typename Tensor_expr::Index_vec;

    // @param filter A functor with the signature
    //        Value_type operator()(const Index_vec&, Value_type)
    Mask_view(const Tensor_expr& expr, const Tensor_expr& mask, Value_type masked_value)
        : Tensor_base<Order>(expr.sizes())
        , m_expr(expr)
        , m_mask(mask)
        , masked_value(masked_value)
    {}

    Value_type operator[](const Index_vec& idx) const {
        return m_filter(m_expr[idx], idx);
    }

    // The ability to write through this view does not make much sense here
    // but currently no concept for read only views exists.
    Value_type& operator[](const Index_vec& idx) {
        return m_expr[idx];
    }

private:
    details::Tensor_expr_t<Tensor_expr> m_expr;
    details::Tensor_expr_t<Tensor_expr> m_mask;
    Value_type m_masked_value;
};

template <typename Tensor_expr, typename Filter>
Filter_view<Tensor_expr, Filter> filter_view(const Tensor_expr& expr, Filter filter) {
    return Filter_view<Tensor_expr, Filter>(expr, filter);
}*/

// A view for the lazy evaluation of Tensor * scalar.
template <typename Tensor_expr, typename Filter>
class Scaling_view : public Tensor_base<Tensor_expr::Order> {
public:
	static constexpr int Order = Tensor_expr::Order;
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = typename Tensor_expr::Index_vec;

	// @param filter A functor with the signature
	//        Value_type operator()(const Index_vec&, Value_type)
	Scaling_view(const Tensor_expr& expr, Value_type scalar)
		: Tensor_base<Order>(expr.sizes())
		, m_expr(expr)
		, m_scalar(scalar) {}

	Value_type operator[](const Index_vec& idx) const {
		return m_expr[idx] * m_scalar;
	}

	// The ability to write through this view does not make much sense here
	// but currently no concept for read only views exists.
	Value_type& operator[](const Index_vec& idx) {
		return m_expr[idx];
	}

private:
	details::Tensor_expr_t<Tensor_expr> m_expr;
	Value_type m_scalar;
};

template <typename Tensor_expr, typename T>
Scaling_view<Tensor_expr, T> scaling_view(const Tensor_expr& expr, T scalar) {
	return Scaling_view<Tensor_expr, T>(expr, scalar);
}

template<typename Tensor_expr, int Dim>
class Layer_view : public Tensor_base<Tensor_expr::Order-1> {
public:
	static constexpr int Order = Tensor_expr::Order-1;
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = Base_index_vec<Order>;

	Layer_view(const Tensor_expr& expr, int layer = 0)
		: Tensor_base<Order>(squeeze_idx(expr.sizes()))
		, m_expr(expr)
		, m_layer(layer) {
			ASSERT(expr.sizes()[Dim] > layer);
		}

	Value_type operator[](const Index_vec& idx) const {
		return m_expr[unsqueeze_idx(idx)];
	}

	Value_type& operator[](const Index_vec& idx) {
		return m_expr[unsqueeze_idx(idx)];
	}

private:
	using Full_index_vec = typename Tensor_expr::Index_vec;

	Index_vec squeeze_idx(const Full_index_vec& full_idx) const {
		Index_vec idx;
		for(int i = 0; i < Order; ++i) {
			idx[i] = full_idx[i < Dim ? i : i+1];
		}
		
		return idx;
	}

	Full_index_vec unsqueeze_idx(const Index_vec& idx) const {
		Full_index_vec full_idx;
		for(int i = 0; i < Order; ++i){
			full_idx[i < Dim ? i : i+1] = idx[i];
		}
		full_idx[Dim] = m_layer;

		return full_idx;
	}

	details::Tensor_expr_t<Tensor_expr> m_expr;
	int m_layer;
};

template<int Dim, typename Tensor_expr>
Layer_view<Tensor_expr, Dim> layer_view(const Tensor_expr& expr, int layer) {
	return Layer_view<Tensor_expr, Dim>(expr, layer);
}

// Basically expr1 = expr2.
template <typename Tensor_expr1, typename Tensor_expr2>
void assign_view(Tensor_expr1& expr1, const Tensor_expr2& expr2);

// ******************************************************************* //
//  Implementation
// ******************************************************************* //
template <typename T, int N, Storage_order S>
Tensor<T, N, S>::Tensor(const Tensor<T, N, S>& oth)
	: Tensor_base<N, S>(oth)
	, m_data(new T[this->num_elem()]) {
	std::copy(oth.m_data.get(), oth.m_data.get() + this->num_elem(), m_data.get());
}

template <typename T, int N, Storage_order S>
Tensor<T, N, S>::Tensor(Tensor<T, N, S>&& oth) noexcept
	: Tensor_base<N, S>(oth)
	, m_data(std::move(oth.m_data)) {
}

template <typename T, int N, Storage_order S>
template <typename Tensor_expr, typename>
Tensor<T, N, S>::Tensor(const Tensor_expr& view)
	: Tensor_base<N, S>(view.sizes())
	, m_data(new T[this->num_elem()]) {
	for (Flat_index flat = 0; flat < this->num_elem(); ++flat) {
		m_data[flat] = view[this->index(flat)];
	}
}

template <typename T, int N, Storage_order S>
Tensor<T, N, S>& Tensor<T, N, S>::operator=(Tensor<T, N, S>&& oth) noexcept {
	Tensor_base<N>::resize(oth.sizes());
	m_data = std::move(oth.m_data);

	return *this;
}

template <typename T, int N, Storage_order S>
Tensor<T, N, S>& Tensor<T, N, S>::operator=(const Tensor<T, N, S>& oth) {
	Tensor_base<N, S>::resize(oth.sizes());
	m_data.reset(new T[this->num_elem()]);
	std::copy_n(oth.data(), this->num_elem(), m_data.get());

	return *this;
}

template <typename T, int N, Storage_order S>
template <typename Tensor_expr, typename>
Tensor<T, N, S>& Tensor<T, N, S>::operator=(const Tensor_expr& view) {
	Tensor_base<N, S>::resize(view.sizes());
	m_data.reset(new T[this->num_elem()]);
	for (Flat_index flat = 0; flat < this->num_elem(); ++flat) {
		m_data[flat] = view[this->index(flat)];
	}

	return *this;
}

template <typename T, int N, Storage_order S>
Tensor<T, N, S> Tensor<T, N, S>::zeros(const Index_vec& sizes) {
	Tensor<T, N> tensor(sizes);
	std::fill_n(tensor.data(), tensor.num_elem(), static_cast<T>(0));

	return tensor;
}

template <typename T, int N, Storage_order S>
Tensor<T, N, S> Tensor<T, N, S>::ones(const Index_vec& sizes) {
	Tensor<T, N> tensor(sizes);
	std::fill_n(tensor.data(), tensor.num_elem(), static_cast<T>(1));

	return tensor;
}

template<typename T, int N, Storage_order S>
void Tensor<T,N,S>::zero() {
	std::fill_n(this->data(), this->num_elem(), static_cast<T>(0));
}

template <int N, Storage_order S>
Flat_index Tensor_base<N, S>::flat_index(const Index_vec& idx) const noexcept  {
	// column major
	/*	std::size_t flatInd = idx[0];
	    std::size_t dimSize = m_sizes[0];
	    for (int i = 1; i < N; ++i) {
	        flatInd += dimSize * idx[i];
	        dimSize *= m_sizes[i];
	    }

	    return flatInd;*/
	// row major
	Flat_index flat_idx = idx[N - 1];
	Flat_index dim_size = m_sizes[N - 1];
	for (int i = N - 2; i >= 0; --i) {
		flat_idx += dim_size * idx[i];
		dim_size *= m_sizes[i];
	}

	return flat_idx;
}

//#define USE_PRECOMPUTED_STRIDES

template <>
inline Flat_index Tensor_base<3, Storage_order::ROW_MAJOR>::flat_index(const Index_vec& idx) const noexcept {
#ifdef USE_PRECOMPUTED_STRIDES
	Flat_index flat_idx = idx[2];
	flat_idx += m_strides[1] * idx[1];
	flat_idx += m_strides[0] * idx[0];
	return flat_idx;
#else
	Flat_index flat_idx = idx[2];
	Flat_index dim_size = m_sizes[2];
	flat_idx += dim_size * idx[1];
	dim_size *= m_sizes[1];
	flat_idx += dim_size * idx[0];
	return flat_idx;
#endif
}

template <>
inline Flat_index Tensor_base<4, Storage_order::ROW_MAJOR>::flat_index(const Index_vec& idx) const noexcept {
#ifdef USE_PRECOMPUTED_STRIDES
	Flat_index flat_idx = idx[3];
	flat_idx += m_strides[2] * idx[2];
	flat_idx += m_strides[1] * idx[1];
	flat_idx += m_strides[0] * idx[0];
	return flat_idx;
#else
	Flat_index flat_idx = idx[3];
	Flat_index dim_size = m_sizes[3];
	flat_idx += dim_size * idx[2];
	dim_size *= m_sizes[2];
	flat_idx += dim_size * idx[1];
	dim_size *= m_sizes[1];
	flat_idx += dim_size * idx[0];
	return flat_idx;
#endif
}

template <>
inline Flat_index Tensor_base<4, Storage_order::SWITCH_LAST>::flat_index(const Index_vec& idx) const noexcept {
	Flat_index flat_idx = idx[2];
	Flat_index dim_size = m_sizes[2];
	flat_idx += dim_size * idx[1];
	dim_size *= m_sizes[1];
	flat_idx += dim_size * idx[0];
	dim_size *= m_sizes[0];
	flat_idx += dim_size * idx[3];
	return flat_idx;
}

template <int N, Storage_order S>
Base_index_vec<N> Tensor_base<N, S>::index(Flat_index flat) const noexcept {
	Index_vec result;

	for (int i = N - 1; i > 0; --i) {
		result[i] = flat % m_sizes[i];
		flat /= m_sizes[i];
	}

	ASSERT(flat < m_sizes.front());
	result.front() = flat;

	return result;
}

template <>
inline Base_index_vec<4> Tensor_base<4, Storage_order::SWITCH_LAST>::index(Flat_index flat) const noexcept {
	Index_vec result;

	for (int i = 2; i >= 0; --i) {
		result[i] = flat % m_sizes[i];
		flat /= m_sizes[i];
	}

	ASSERT(flat < m_sizes.back());
	result.back() = flat;

	return result;
}

template <int N, Storage_order S>
bool Tensor_base<N, S>::is_valid_idx(const Index_vec& idx) const noexcept {
	for (int i = 0; i < N; ++i)
		if (idx[i] < 0 || idx[i] >= m_sizes[i]) {
			std::cout << idx << "\n";
			return false;
		}
	return true;
}

template <int N, Storage_order S>
Flat_index Tensor_base<N, S>::compute_num_elem(const Index_vec& sizes) noexcept {
	Flat_index s = sizes[0];
	for (int i = 1; i < N; ++i) {
		s *= static_cast<Flat_index>(sizes[i]);
	}
	return s;
}

template <int N, Storage_order S>
Base_index_vec<N> Tensor_base<N,S>::compute_strides(const Index_vec& sizes) noexcept {
	Index_vec strides;
	strides[N-1] = 1;
	for(int i = N-2; i >= 0; --i)
		strides[i] = strides[i+1] * sizes[i+1];

	return strides;
}

template<>
inline Base_index_vec<4> Tensor_base<4,Storage_order::SWITCH_LAST>::compute_strides(const Index_vec& sizes) noexcept {
	Index_vec strides;
	strides[2] = 1;
	for(int i = 1; i >= 0; --i)
		strides[i] = strides[i+1] * sizes[i+1];

	strides[3] = strides[0] * sizes[0];
	return strides;
}

template <int N, Storage_order S>
Flat_index Tensor_base<N,S>::flat_index(Flat_index base, Index offset) const noexcept {
	return base * m_sizes[N-1] + offset;
}

template <>
inline Flat_index Tensor_base<4, Storage_order::SWITCH_LAST>::flat_index(Flat_index base, Index offset) const noexcept {
	return base + m_strides[3] * offset;
}

// TENSOR
template <typename T, int N, Storage_order S>
Tensor<T, N, S>& Tensor<T, N, S>::operator*=(T scalar) {
	for (Flat_index i = 0; i < this->num_elem(); ++i) {
		m_data[i] *= scalar;
	}
	return *this;
}

template <typename T, int N, Storage_order S>
Tensor<T, N, S>& Tensor<T, N, S>::operator+=(const Tensor<T, N, S>& oth) {
	ASSERT(this->sizes() == oth.sizes());
	for (Flat_index i = 0; i < this->num_elem(); ++i) {
		m_data[i] += oth.m_data[i];
	}
	return *this;
}

template <typename T, int N, Storage_order S>
bool Tensor<T, N, S>::operator==(const Tensor<T, N, S>& oth) const {
	for (Flat_index i = 0; i < this->num_elem(); ++i) {
		if (m_data[i] != oth[i])
			return false;
	}
	return true;
}

// VIEWS
template <typename Tensor_expr1, typename Tensor_expr2>
void assign_view(Tensor_expr1& expr1, const Tensor_expr2& expr2) {
	static_assert(Tensor_expr1::Order == Tensor_expr2::Order, "Order must be the same for assignment.");
	ASSERT(expr1.sizes() == expr2.sizes());
	const Flat_index num_elem = expr1.num_elem();
	for (Flat_index flat = 0; flat < num_elem; ++flat) {
		const auto idx = expr1.index(flat);
		expr1[idx] = expr2[idx];
	}
}

#endif