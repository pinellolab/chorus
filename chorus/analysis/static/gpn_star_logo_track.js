(function () {
    const BASE_ORDER = ["A", "C", "G", "T"];
    const BASE_COLORS = { A: "#109648", C: "#255C99", G: "#F7B32B", T: "#D62839" };
    const MAX_STACK_HEIGHT = 2.0;

    // fillText is expensive (font shaping + rasterization) compared to
    // fillRect, and at the full ~100,000bp conservation window that's up
    // to 4 calls x 100,001 positions per redraw -- every pan/zoom. Below
    // this zoom, each position is sub-pixel wide anyway (illegible as
    // text regardless), so fall back to cheap colored bars. Matches the
    // threshold igv.js's own native "dynseq" wig graph type uses for the
    // same reason (draws letters only when bpPerPixel < 2).
    const LETTER_ZOOM_THRESHOLD = 2;

    // Reference font size used purely to measure each glyph's natural
    // proportions; the actual on-screen size comes from the scale()
    // transform applied per-segment in drawLetter.
    const GLYPH_REF_SIZE = 100;
    const GLYPH_FONT = `bold ${GLYPH_REF_SIZE}px Arial, "Helvetica Neue", sans-serif`;

    // measureText is comparatively expensive and A/C/G/T never change
    // shape, so cache their metrics after the first measurement.
    const glyphMetricsCache = {};

    function getGlyphMetrics(context, base) {
        if (glyphMetricsCache[base]) return glyphMetricsCache[base];
        context.save();
        context.font = GLYPH_FONT;
        const m = context.measureText(base);
        // actualBoundingBox* gives the tight glyph box (not the font's
        // generic ascent/descent), so A/C/G/T each get scaled from their
        // real ink extents rather than a shared, looser box that would
        // make some letters look small/off-center.
        const metrics = {
            ascent: m.actualBoundingBoxAscent,
            descent: m.actualBoundingBoxDescent,
            width: m.actualBoundingBoxRight - m.actualBoundingBoxLeft,
            left: m.actualBoundingBoxLeft
        };
        context.restore();
        glyphMetricsCache[base] = metrics;
        return metrics;
    }

    // Draws `base` stretched to exactly fill the box (x, y, w, h).
    function drawLetter(context, base, x, y, w, h, color) {
        const { ascent, descent, width, left } = getGlyphMetrics(context, base);
        const glyphHeight = ascent + descent;
        if (glyphHeight <= 0 || width <= 0 || w <= 0 || h <= 0) return;

        const scaleX = w / width;
        const scaleY = h / glyphHeight;

        context.save();
        context.fillStyle = color;
        context.font = GLYPH_FONT;
        // Move origin to the box's top-left, scale so one reference-size
        // glyph unit maps onto (w, h), then draw the glyph offset so its
        // tight bounding box lands at (0,0)-(width,glyphHeight) in that
        // scaled space. fillText's y-arg is the alphabetic baseline, which
        // sits `ascent` below the ink's top -- passing `ascent` here puts
        // the ink's top at local y=0 and its bottom at local y=glyphHeight,
        // which map to global y and y+h respectively once scaled. An
        // earlier version translated to (x, y + h) instead of (x, y): that
        // shifts the whole glyph an extra `h` further down (ink ends up at
        // global [y+h, y+2h]), invisible for tall letters since y+2h lands
        // off the bottom of the track -- exactly why only the short bases
        // in a stack were ever visible, never the tallest/dominant one.
        context.translate(x, y);
        context.scale(scaleX, scaleY);
        context.fillText(base, -left, ascent);
        context.restore();
    }

    class GpnStarStackedLogoTrack extends igv.TrackBase {
        constructor(config, browser) {
            super(config, browser);
        }

        init(config) {
            super.init(config);
            this.type = "gpnstarstackedlogo";
            this.featureList = config.features || [];
        }

        async getFeatures(chr, start, end) {
            return this.featureList.filter(f => f.chr === chr && f.end > start && f.start < end);
        }

        computePixelHeight() {
            return this.height;
        }

        draw(options) {
            const { features, context, bpPerPixel, bpStart, pixelHeight } = options;
            if (!features || features.length === 0) {
                return;
            }

            const drawLetters = bpPerPixel < LETTER_ZOOM_THRESHOLD;

            for (const f of features) {
                const x = (f.start - bpStart) / bpPerPixel;
                const w = Math.max(1, (f.end - f.start) / bpPerPixel);

                // Classic sequence-logo stacking order ("big_on_top", the
                // Logomaker/WebLogo default): tallest letter farthest from
                // the baseline (at the top), decreasing in height going
                // down — NOT the fixed A/C/G/T order, which can put a small
                // letter above a much taller one. Ascending sort here since
                // the loop below walks the baseline upward and draws last,
                // so the largest (drawn last) ends up at the top.
                const order = BASE_ORDER
                    .slice()
                    .sort((a, b) => (f["p" + a] || 0) - (f["p" + b] || 0));

                let y = pixelHeight;
                for (const base of order) {
                    const h = ((f["p" + base] || 0) / MAX_STACK_HEIGHT) * pixelHeight;
                    if (h <= 0) {
                        continue;
                    }
                    y -= h;
                    if (drawLetters) {
                        drawLetter(context, base, x, y, w, h, BASE_COLORS[base]);
                    } else {
                        context.fillStyle = BASE_COLORS[base];
                        context.fillRect(x, y, w, h);
                    }
                }
            }
        }

        menuItemList() {
            return [];
        }
    }

    igv.registerTrackClass("gpnstarstackedlogo", GpnStarStackedLogoTrack);
})();