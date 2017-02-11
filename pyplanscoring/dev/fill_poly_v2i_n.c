/* C99, public domain licensed */

#include <limits.h>
#include <stdbool.h>
#include <math.h>

/* utilities */

#define SWAP(type, a, b) do {  \
	type sw_ap;                \
	sw_ap = (a);               \
	(a) = (b);                 \
	(b) = sw_ap;               \
} while (0)

static inline int min_ii(int a, int b)
{
	return (a < b) ? a : b;
}

static inline int max_ii(int a, int b)
{
	return (b < a) ? a : b;
}

/* sort edge-segments on y, then x axis */
static int fill_poly_v2i_n__span_y_sort(const void *a_p, const void *b_p, void *verts_p)
{
	const int (*verts)[2] = verts_p;
	const int *a = a_p;
	const int *b = b_p;
	const int *co_a = verts[a[0]];
	const int *co_b = verts[b[0]];

	if (co_a[1] < co_b[1]) {
		return -1;
	}
	else if (co_a[1] > co_b[1]) {
		return 1;
	}
	else if (co_a[0] < co_b[0]) {
		return -1;
	}
	else if (co_a[0] > co_b[0]) {
		return 1;
	}
	else {
		/* co_a & co_b are identical, use the line closest to the x-min */
		const int *co = co_a;
		co_a = verts[a[1]];
		co_b = verts[b[1]];
		int ord = (((co_b[0] - co[0]) * (co_a[1] - co[1])) -
		           ((co_a[0] - co[0]) * (co_b[1] - co[1])));
		if (ord > 0) {
			return -1;
		}
		if (ord < 0) {
			return 1;
		}
	}
	return 0;
}

/**
 * \param callback: Takes the x, y coords and x-span (\a x_end is not inclusive),
 * note that \a x_end will always be greater than \a x, so we can use:
 *
 * \code{.c}
 * do {
 *     func(x, y);
 * } while (++x != x_end);
 * \endcode
 */
void fill_poly_v2i_n(
        const int xmin, const int ymin, const int xmax, const int ymax,
        const int verts[][2], const int nr,
        void (*callback)(int x, int x_end, int y, void *), void *user_data)
{
	/* Originally by Darel Rex Finley, 2007.
	 * Optimized by Campbell Barton, 2016 to keep sorted intersections. */

	int (*span_y)[2] = malloc(sizeof(*span_y) * (size_t)nr);
	int span_y_len = 0;

	for (int i_curr = 0, i_prev = nr - 1; i_curr < nr; i_prev = i_curr++) {
		const int *co_prev = verts[i_prev];
		const int *co_curr = verts[i_curr];

		if (co_prev[1] != co_curr[1]) {
			/* Any segments entirely above or below the area of interest can be skipped. */
			if ((min_ii(co_prev[1], co_curr[1]) >= ymax) ||
			    (max_ii(co_prev[1], co_curr[1]) <  ymin))
			{
				continue;
			}

			int *s = span_y[span_y_len++];
			if (co_prev[1] < co_curr[1]) {
				s[0] = i_prev;
				s[1] = i_curr;
			}
			else {
				s[0] = i_curr;
				s[1] = i_prev;
			}
		}
	}

	qsort_r(span_y, (size_t)span_y_len, sizeof(*span_y), fill_poly_v2i_n__span_y_sort, (void *)verts);

	struct NodeX {
		int span_y_index;
		int x;
	} *node_x = malloc(sizeof(*node_x) * (size_t)nr, __func__);
	int node_x_len = 0;

	int span_y_index = 0;
	if (span_y_len != 0 && verts[span_y[0][0]][1] < ymin) {
		while ((span_y_index < span_y_len) &&
		       (verts[span_y[span_y_index][0]][1] < ymin))
		{
			assert(verts[span_y[span_y_index][0]][1] <
			       verts[span_y[span_y_index][1]][1]);
			if (verts[span_y[span_y_index][1]][1] >= ymin) {
				struct NodeX *n = &node_x[node_x_len++];
				n->span_y_index = span_y_index;
			}
			span_y_index += 1;
		}
	}

	/* Loop through the rows of the image. */
	for (int pixel_y = ymin; pixel_y < ymax; pixel_y++) {
		bool is_sorted = true;
		bool do_remove = false;

		for (int i = 0, x_ix_prev = INT_MIN; i < node_x_len; i++) {
			struct NodeX *n = &node_x[i];
			const int *s = span_y[n->span_y_index];
			const int *co_prev = verts[s[0]];
			const int *co_curr = verts[s[1]];

			assert(co_prev[1] < pixel_y && co_curr[1] >= pixel_y);

			const double x    = (co_prev[0] - co_curr[0]);
			const double y    = (co_prev[1] - co_curr[1]);
			const double y_px = (pixel_y    - co_curr[1]);
			const int    x_ix = (int)round((double)co_curr[0] + ((y_px / y) * x));
			n->x = x_ix;

			if (is_sorted && (x_ix_prev > x_ix)) {
				is_sorted = false;
			}
			if (do_remove == false && co_curr[1] == pixel_y) {
				do_remove = true;
			}
			x_ix_prev = x_ix;
		}

		/* Sort the nodes, via a simple "Bubble" sort. */
		if (is_sorted == false) {
			int i = 0;
			const int current_end = node_x_len - 1;
			while (i < current_end) {
				if (node_x[i].x > node_x[i + 1].x) {
					SWAP(struct NodeX, node_x[i], node_x[i + 1]);
					if (i != 0) {
						i -= 1;
					}
				}
				else {
					i += 1;
				}
			}
		}

		/* Fill the pixels between node pairs. */
		for (int i = 0; i < node_x_len; i += 2) {
			int x_src = node_x[i].x;
			int x_dst = node_x[i + 1].x;

			if (x_src >= xmax) {
				break;
			}

			if (x_dst > xmin) {
				if (x_src < xmin) {
					x_src = xmin;
				}
				if (x_dst > xmax) {
					x_dst = xmax;
				}
				/* for single call per x-span */
				if (x_src < x_dst) {
					callback(x_src - xmin, x_dst - xmin, pixel_y - ymin, user_data);
				}
			}
		}

		/* Clear finalized nodes in one pass, only when needed
		 * (avoids excessive array-resizing). */
		if (do_remove == true) {
			int i_dst = 0;
			for (int i_src = 0; i_src < node_x_len; i_src += 1) {
				const int *s = span_y[node_x[i_src].span_y_index];
				const int *co = verts[s[1]];
				if (co[1] != pixel_y) {
					if (i_dst != i_src) {
						/* x is initialized for the next pixel_y (no need to adjust here) */
						node_x[i_dst].span_y_index = node_x[i_src].span_y_index;
					}
					i_dst += 1;
				}
			}
			node_x_len = i_dst;
		}

		/* scan for new x-nodes */
		while ((span_y_index < span_y_len) &&
		       (verts[span_y[span_y_index][0]][1] == pixel_y))
		{
			/* note, node_x these are just added at the end,
			 * not ideal but sorting once will resolve. */

			/* x is initialized for the next pixel_y */
			struct NodeX *n = &node_x[node_x_len++];
			n->span_y_index = span_y_index;
			span_y_index += 1;
		}
	}

	free(span_y);
	free(node_x);
}

