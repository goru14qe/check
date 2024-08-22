#ifndef VEC_H
#define VEC_H

#include <array>

using Vec3 = std::array<double, 3>;

inline double dot(const Vec3& v, const Vec3& u) {
	return v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
}

inline Vec3 operator+(const Vec3& u, const Vec3& v) {
	return Vec3{u[0] + v[0], u[1] + v[1], u[2] + v[2]};
}

inline Vec3 operator-(const Vec3& u, const Vec3& v) {
	return Vec3{u[0] - v[0], u[1] - v[1], u[2] - v[2]};
}

inline Vec3 operator*(const Vec3& u, double s) {
	return Vec3{s * u[0], s * u[1], s * u[2]};
}

inline Vec3 operator*(double s, const Vec3& u) {
	return Vec3{s * u[0], s * u[1], s * u[2]};
}

inline Vec3 operator/(const Vec3& u, double s) {
	return Vec3{u[0] / s, u[1] / s, u[2] / s};
}

inline Vec3& operator+=(Vec3& slf, const Vec3& oth) {
	slf[0] += oth[0];
	slf[1] += oth[1];
	slf[2] += oth[2];
	return slf;
}

inline Vec3& operator*=(Vec3& slf, double s) {
	slf[0] *= s;
	slf[1] *= s;
	slf[2] *= s;
	return slf;
}

inline Vec3 operator-(const Vec3& u){
	return Vec3{-u[0], -u[1], -u[2]};
}

#endif