export interface ImageReport {
    logits: number;
    prediction_class: number;
    prediction_label: string;
    probability_fake: number;
    status: string;
}