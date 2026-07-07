#!/bin/bash
set -euo pipefail

JOBS=""

apply_patch()
{
	local dir="$1"
	local patch_file="$2"
	if patch -d "$dir" -p1 --dry-run --reverse --force < "$patch_file" >/dev/null 2>&1; then
		printf '\033[1;33m==> Patch already applied, skipping: %s\033[0m\n' "$patch_file"
		return 0
	fi
	patch -d "$dir" -p1 < "$patch_file"
}

build_kleidicv()
{
	printf '\033[1;34m==> Build started: kleidicv\033[0m\n'
	cmake -S kleidicv -B build-kleidicv -DCMAKE_BUILD_TYPE=Release -DKLEIDICV_BENCHMARK=ON
	cmake --build build-kleidicv -j ${JOBS}
	printf '\033[1;32m==> Build finished: kleidicv\033[0m\n'
}

build_opencv()
{
	local dir="$1"
	local main_patch="$2"
	local benchmark_patch="$3"
	shift 3
	apply_patch "$dir" "$main_patch"
	apply_patch "$dir" "$benchmark_patch"
	printf '\033[1;34m==> Build started: %s\033[0m\n' "$dir"
	cmake -S "$dir" -B "build-$dir" -DBUILD_TESTS=OFF -DWITH_KLEIDICV=OFF
	cmake --build "build-$dir" -j ${JOBS}
	printf '\033[1;32m==> Build finished: %s\033[0m\n' "$dir"
	printf '\033[1;34m==> Build started: %s-kcv\033[0m\n' "$dir"
	cmake -S "$dir" -B "build-$dir-kcv" -DBUILD_TESTS=OFF -DWITH_KLEIDICV=ON -DKLEIDICV_SOURCE_PATH="$(pwd)/kleidicv" "$@"
	cmake --build "build-$dir-kcv" -j ${JOBS}
	printf '\033[1;32m==> Build finished: %s-kcv\033[0m\n' "$dir"
}

build_opencv_4()
{
	build_opencv opencv-4 \
		kleidicv/adapters/opencv/opencv-4.13.patch \
		kleidicv/adapters/opencv/extra_benchmarks/opencv-4.13.patch \
		"$@"
}

build_opencv_5()
{
	build_opencv opencv-5 \
		kleidicv/adapters/opencv/opencv-5.x.patch \
		patch/extra_benchmarks-opencv-5.x.patch \
		"$@"
}

usage()
{
	cat <<'EOF'
Usage: build.sh [command] [options]

Commands:
  help       Show this help message and exit
  build      Default build of KleidiCV, OpenCV 4 and OpenCV 5
             (w/o and with KleidiCV enabled)

Options (for build):
  all_hal    Build with all OpenCV HAL entry points enabled
             (-DKLEIDICV_ENABLE_ALL_OPENCV_HAL=ON)
  -j N       Number of parallel build jobs (default: all cores)

EOF
}

case "${1:-}" in
	build)
		shift
		extra_cmake_args=()
		while [ $# -gt 0 ]; do
			case "$1" in
				all_hal)
					extra_cmake_args=(-DKLEIDICV_ENABLE_ALL_OPENCV_HAL=ON)
					;;
				-j)
					if [ -z "${2:-}" ]; then
						printf 'Option -j requires an argument\n\n' >&2
						usage >&2
						exit 1
					fi
					JOBS="$2"
					shift
					;;
				-j*)
					JOBS="${1#-j}"
					;;
				*)
					printf 'Unknown option: %s\n\n' "$1" >&2
					usage >&2
					exit 1
					;;
			esac
			shift
		done
		build_kleidicv
		build_opencv_4 "${extra_cmake_args[@]}"
		build_opencv_5 "${extra_cmake_args[@]}"
		printf '\033[1;32m==> All builds finished\033[0m\n'
		;;
	*)
		usage
		;;
esac
