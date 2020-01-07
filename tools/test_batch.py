import numpy as np
import multiprocessing
import argparse
import datetime
import time
import tqdm 
import sys
import os

from collections import defaultdict
from lvis import LVIS, LVISEval, LVISResults


class CustomLVISEval(LVISEval):
    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_imgs:
            self.logger.warn("Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones(
            (num_thrs, num_recalls, num_cats, num_area_rngs)
        )
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [
                    self.eval_imgs[Nk + Na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                # Remove elements which are None
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                dt_pointers[cat_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "dt_scores": dt_scores,
                    "tps": tps,
                    "fps": fps,
                }

                if num_gt == 0:
                    continue

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[
                            -1
                        ]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(
                        rc, self.params.rec_thrs, side="left"
                    )

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except:
                        pass
                    precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)

        self.eval = {
            "params": self.params,
            "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
        }


def lvis_eval(lvis, lvis_results, category_id, score_threshold=0.0):
    assert isinstance(lvis, LVIS)

    lvis_results_start = time.time()
    img_ids = lvis.get_img_ids()
    res_type = 'bbox'
    iou_type = 'bbox' if res_type == 'proposal' else res_type
    lvis_eval = CustomLVISEval(lvis, lvis_results, iou_type)
    cat_ids = [category_id]
    lvis_eval.params.cat_ids = cat_ids
    lvis_eval.run()
    lvis_eval.print_results()
    # Image level accuracy
    im_start = time.time()
    dt_map = {det['id']: image_id
              for (image_id, cat_id), dets in lvis_eval._dts.items()
              for det in dets}
    num_area_rngs = len(lvis_eval.params.area_rng)
    num_imgs = len(lvis_eval.params.img_ids)
    image_tps = defaultdict(int)
    image_fps = defaultdict(int)
    for cat_idx in range(len(lvis_eval.params.cat_ids)):
        Nk = cat_idx * num_area_rngs * num_imgs
        for area_idx in range(num_area_rngs):
            d = lvis_eval.eval['dt_pointers'][cat_idx][area_idx]
            if not d:
                continue
            image_ids = [dt_map[d_id] for d_id in d['dt_ids']]
            dt_scores = d['dt_scores']
            tps = d['tps']
            fps = d['fps']
            print('target score', score_threshold)
            for idx, img_id in enumerate(image_ids):
                print('dt score', dt_scores[idx])
                if dt_scores[idx] < score_threshold:
                    continue
                if tps[:,idx].any():
                    image_tps[img_id] += 1
                if fps[:,idx].all():
                    image_fps[img_id] += 1
    correct_ids = []
    incorrect_ids = []
    unsure_ids = []
    total = num_imgs
    num_dets = sum([len(v) for k, v in lvis_eval._dts.items()])
    num_det_imgs = sum([1 for k, v in lvis_eval._dts.items()
                       if len(v) > 0])
    num_gts = sum([len(v) for (_, cat_id), v in lvis_eval._gts.items()
                   if cat_id in cat_ids])
    fn_gt_img_ids = set(
        [img_id for (img_id, cat_id), v in lvis_eval._gts.items()
         if cat_id in cat_ids and len(v) > 0])
    num_gt_imgs = len(fn_gt_img_ids)
    for image_id in lvis_eval.params.img_ids:
        is_tp = image_tps[image_id] > 0
        is_fp = image_fps[image_id] > 0
        if is_tp:
            correct_ids.append(image_id)
            fn_gt_img_ids.remove(image_id)
        elif is_fp:
            incorrect_ids.append(image_id)
        else:
            unsure_ids.append(image_id)
    print('Image recall')
    print('TPs: {:s}'.format(','.join([str(x) for x in correct_ids])))
    print('FPs: {:s}'.format(','.join([str(x) for x in incorrect_ids])))
    print('FNs: {:s}'.format(','.join([str(x) for x in fn_gt_img_ids])))
    print('num dts: {:d}'.format(num_dets))
    print('num gts: {:d}'.format(num_gts))
    print('num dt imgs: {:d}'.format(num_det_imgs))
    print('num gt imgs: {:d}'.format(num_gt_imgs))


GLOBAL_SYN = None
GLOBAL_RESULTS = {}


def eval_single(data):
    global GLOBAL_SYN
    global GLOBAL_RESULTS
    category, json_file_path, score_threshold, results_path, save_path, text_output = data
    sys.stdout = open(text_output, 'w')

    score_threshold = float(score_threshold)

    lvis_api = LVIS(json_file_path)
    lvis_api.load_cats(None)

    if not GLOBAL_SYN:
        syn_to_id = {}
        for cat_id, cat_data in lvis_api.cats.items():
            syn_to_id[cat_data['synonyms'][0]] = cat_id
        GLOBAL_SYN = syn_to_id

    syn_to_id = GLOBAL_SYN
    category_id = syn_to_id[category]

    if results_path not in GLOBAL_RESULTS:
        lvis_results = LVISResults(lvis_api, results_path)
        GLOBAL_RESULTS[results_path] = lvis_results

    lvis_results = GLOBAL_RESULTS[results_path]

    lvis_eval(lvis_api, lvis_results, category_id, score_threshold=score_threshold)
    sys.stdout.close()


def main():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('batch_file', help='path to files')
    args = parser.parse_args()

    with open(args.batch_file, 'r') as f:
        lines = f.readlines()
        eval_tuples = [line.strip().split(',') for line in lines]

    pool = multiprocessing.Pool()

    for _ in tqdm.tqdm(pool.imap_unordered(eval_single, eval_tuples), total=len(eval_tuples)):
        pass


if __name__ == '__main__':
        main()
