module;

#include <cmath>

export module Corr;

export struct Corr {
    int x, y;

    Corr() = default;
    Corr(const Corr& c) = default;
    Corr(Corr&& c) = default;
    ~Corr() = default;

    Corr(int x, int y) : x(x), y(y) {}


    inline bool operator<(const Corr& c) const {
        if (this->x != c.x)
            return this->x < c.x;
        else
            return this->y < c.y;
    }

    inline bool operator>(const Corr& c) const {
        if (this->x != c.x)
            return this->x > c.x;
        else
            return this->y > c.y;
    }

    inline bool operator==(const Corr& c) const {
        return (this->x == c.x && this->y == c.y);
    }

    inline Corr operator+(const Corr& c) const {
        return Corr(this->x + c.x, this->y + c.y);
    }

    inline Corr operator-(const Corr& c) const {
        return Corr(this->x - c.x, this->y - c.y);
    }
};

export template <typename T>
struct CorrWithVal {

    Corr corr;
    T val;

    CorrWithVal() = default;
    CorrWithVal(const CorrWithVal&) = default;
    CorrWithVal(CorrWithVal&&) = default;
    ~CorrWithVal() = default;

    CorrWithVal(const T& val, const Corr& c) : corr(c), val(val) {}
    CorrWithVal(const T& val, const int x, const int y) : corr(x, y), val(val) {}

    inline bool operator<(const CorrWithVal& c) const {
        if (this->val != c.val)
            return this->val < c.val;
        else
            return this->corr < c.corr;
    }

    inline bool operator>(const CorrWithVal& c) const {
        if (this->val != c.val)
            return this->val < c.val;
        else
            return this->corr < c.corr;
    }

    bool operator==(const CorrWithVal& c) const {
        return this->corr == c.corr && this->val == c.val;
    }

    inline Corr operator+(const CorrWithVal& c) const {
        return Corr(this->corr + c.corr, this->val + c.val);
    }

    inline Corr operator-(const CorrWithVal& c) const {
        return Corr(this->corr - c.corr, this->val - c.val);
    }
};

export inline auto corr_norm(const Corr& c) noexcept {
    return sqrt(c.x * c.x + c.y * c.y);
}

export template<typename T>
inline auto corr_norm(const CorrWithVal<T>& c) noexcept {
    return corr_norm(c.corr);
}