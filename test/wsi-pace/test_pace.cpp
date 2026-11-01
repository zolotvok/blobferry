// SPDX-License-Identifier: GPL-3.0-or-later
// PC unit test of the WSI layer's MALI_WSI_PACE window (K-WP1, wsi/mali_pace.hpp from
// bridge/patches/vulkan-wsi-layer/0002-blobferry-wsi-pace.patch). Run: bridge/test/wsi-pace/run.sh
// [keeper, Claude Opus 5.5] 2026-11-01
#include "mali_pace.hpp"
#include <cassert>
#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace wsi::mali_pace;
static int fails = 0;
#define CHECK(c)                                                                                                       \
   do                                                                                                                  \
   {                                                                                                                   \
      if (!(c))                                                                                                        \
      {                                                                                                                \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                                  \
         fails++;                                                                                                      \
      }                                                                                                                \
   } while (0)
static bool has(const std::string &s, const char *needle) { return s.find(needle) != std::string::npos; }
static const int64_t MS = 1000000;

int main(int argc, char **argv)
{
   const bool expect_on = argc > 1 && std::string(argv[1]) == "on";
   CHECK(enabled() == expect_on);
   if (!expect_on)
   {
      // disabled: add() must not touch the state, nothing is written
      add(M_ACQUIRE_BLOCK, 5 * MS);
      printf("%s\n", fails ? "FAILED" : "PASS (disabled)");
      return fails ? 1 : 0;
   }

   // 1. FIFO-like alternation at 90 Hz: 11.11 / 22.22 ms
   {
      pace_window w;
      for (int i = 0; i < 100; i++)
         w.add(M_PRESENT_INTERVAL, i % 2 ? 22222222 : 11111111);
      std::string t = w.format("12:00:00", 1.666, 1000.0 / 90.0);
      fputs(t.c_str(), stdout);
      CHECK(has(t, "presents 100 (60.0 fps)"));
      CHECK(has(t, "1x 50% 2x 50% 3x 0% 4x 0% off 0%"));
      CHECK(has(t, "hist_ms 11:50 22:50"));
      CHECK(has(t, "p50   11.11"));
      CHECK(has(t, "max   22.22"));
      CHECK(!has(t, "commit_interval")); // no samples, no line
   }
   // 2. nearest-rank percentiles on 1..100 ms, overflow bin, off-vsync share
   {
      pace_window w;
      for (int i = 100; i >= 1; i--)
         w.add(M_COMMIT_INTERVAL, i * MS);
      std::string t = w.format("12:00:05", 5.0, 1000.0 / 90.0);
      fputs(t.c_str(), stdout);
      CHECK(has(t, "p50   50.00 p90   90.00 p99   99.00 max  100.00"));
      CHECK(has(t, ">=50:51"));
      CHECK(has(t, " 1:1 2:1"));
      CHECK(has(t, "mean   50.50"));
   }
   // 3. counts metric and negative clamp; sample cap keeps n exact
   {
      pace_window w;
      for (int i = 0; i < 5000; i++)
         w.add(M_PENDING_DEPTH, (i % 4) * MS);
      w.add(M_FLIP_GPU_WAIT, -5);
      CHECK(w.samples[M_PENDING_DEPTH].size() == MAX_SAMPLES);
      std::string t = w.format("12:00:10", 5.0, 1000.0 / 120.0);
      CHECK(has(t, "pending_depth     n  5000 mean    1.50"));
      CHECK(has(t, "max    3.00"));
      CHECK(has(t, "flip_gpu_wait     n     1 mean    0.00"));
      w.clear();
      CHECK(w.format("x", 5.0, 10.0).find("present_interval") == std::string::npos);
   }
   // 4. end to end through the global state: 6 s of 16.67-ms presents writes one window to MALI_WSI_PACE_FILE
   {
      const char *path = getenv("MALI_WSI_PACE_FILE");
      CHECK(path != nullptr);
      unlink(path);
      int64_t t0 = 1000 * MS;
      for (int i = 0; i <= 360; i++)
      {
         int64_t now = t0 + i * 16666667LL;
         add(M_ACQUIRE_BLOCK, 2 * MS);
         on_present(now, 2);
         on_commit(now + 3 * MS);
      }
      std::ifstream f(path);
      std::stringstream ss;
      ss << f.rdbuf();
      std::string t = ss.str();
      fputs(t.c_str(), stdout);
      CHECK(has(t, "=== "));
      CHECK(has(t, "presents 300 (60.0 fps)")); // window = 300 intervals in 5.0 s
      CHECK(has(t, "hist_ms 16:300"));
      CHECK(has(t, "vsync 16.67ms")); // MALI_WSI_PACE_HZ=60 from run.sh
      CHECK(has(t, "acquire_block     n   301"));
      CHECK(has(t, "pending_depth"));
   }
   printf("%s\n", fails ? "FAILED" : "PASS");
   return fails ? 1 : 0;
}
