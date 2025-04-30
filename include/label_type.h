#ifndef LABEL_TYPE_H
#define LABEL_TYPE_H


struct Coco17Result
{
    int classIndex;
    float top_left_x;
    float top_left_y;
    float width;
    float height;
    Coco17Result()
        : classIndex(-1), top_left_x(-1), top_left_y(-1), width(-1), height(-1) {}

    Coco17Result(int cls, float x, float y, float w, float h)
        : classIndex(cls), top_left_x(x), top_left_y(y), width(w), height(h) {}
};

struct Coco17DetectionResult : Coco17Result
{
    float confidence =-1;
    Coco17DetectionResult()
        : Coco17Result(), confidence(-1) {}

    Coco17DetectionResult(int cls, float x, float y, float w, float h, float conf)
        : Coco17Result(cls, x, y, w, h), confidence(conf) {}
};
#endif // LABEL_TYPE_H







