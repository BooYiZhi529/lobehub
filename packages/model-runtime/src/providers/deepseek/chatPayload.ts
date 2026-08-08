import type Anthropic from '@anthropic-ai/sdk';
import { deepseek as deepseekChatModels, ModelProvider } from 'model-bank';
import type OpenAI from 'openai';

import { buildDefaultAnthropicPayload } from '../../core/anthropicCompatibleFactory';
import type { ChatStreamPayload } from '../../types';
import { getModelPropertyWithFallback } from '../../utils/getFallbackModelProperty';
import { isDeepSeekV4FamilyModel } from '../../utils/modelParse';
import { resolveSafeMaxTokens } from '../../utils/resolveSafeMaxTokens';
import { sanitizeAnthropicThinkingParts } from '../../utils/sanitizeAnthropicThinkingParts';
import { sanitizeDeepSeekJsonPayload } from './sanitizePayload';

export const isDeepSeekV4Model = (model: string | undefined) =>
  isDeepSeekV4FamilyModel(model);

const isEmptyContent = (content: unknown) =>
  content === '' || content === null || content === undefined;

// Require non-empty text: an empty-string `thinking` has never been validated
// against DeepSeek — empty reasoning falls back to the proven `' '` placeholder.
const hasReasoningContent = (reasoning: any) =>
  typeof reasoning?.content === 'string' && reasoning.content !== '';

const buildThinkingBlock = (reasoning: any) =>
  hasReasoningContent(reasoning)
    ? { thinking: reasoning.content, type: 'thinking' as const }
    : undefined;

const toContentArray = (content: any) =>
  Array.isArray(content)
    ? content
    : [{ text: isEmptyContent(content) ? ' ' : content, type: 'text' as const }];

const shouldEnableDeepSeekThinking = (payload: ChatStreamPayload) => {
  if (payload.model === 'deepseek-reasoner') return true;

  return isDeepSeekV4Model(payload.model) && payload.thinking?.type !== 'disabled';
};

const resolveDeepSeekThinking = (
  payload: ChatStreamPayload,
): ChatStreamPayload['thinking'] => {
  if (payload.model === 'deepseek-reasoner') {
    return {
      budget_tokens: payload.thinking?.budget_tokens ?? 1024,
      type: 'enabled',
    };
  }

  if (isDeepSeekV4Model(payload.model)) {
    if (payload.thinking?.type === 'disabled') {
      return {
        budget_tokens: 0,
        type: 'disabled',
      };
    }

    return {
      budget_tokens: payload.thinking?.budget_tokens ?? 1024,
      type: 'enabled',
    };
  }

  if (
    payload.thinking?.type === 'enabled' &&
    payload.thinking.budget_tokens === undefined
  ) {
    return {
      budget_tokens: 1024,
      type: 'enabled',
    };
  }

  return payload.thinking;
};

/**
 * DeepSeek's Anthropic-compatible API uses Anthropic content blocks for assistant
 * reasoning history. For V4 thinking mode we keep an explicit placeholder block
 * so follow-up tool-call turns preserve the same reasoning-history guarantee as
 * the OpenAI-compatible API.
 *
 * @see https://api-docs.deepseek.com/guides/anthropic_api
 * @see https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
 */
const normalizeMessagesForAnthropic = (
  messages: ChatStreamPayload['messages'],
  forceThinking = false,
) =>
  messages.map((message: any) => {
    if (message.role !== 'assistant') return message;

    const { reasoning, ...rest } = message;

    // Array content may already carry thinking parts built by the context
    // engine — sanitize them for DeepSeek instead of stacking another block.
    const existingParts = Array.isArray(message.content)
      ? sanitizeAnthropicThinkingParts(message.content)
      : undefined;

    const hasThinkingPart = existingParts?.some(
      (part: any) => part.type === 'thinking',
    );

    const thinkingBlock = hasThinkingPart
      ? undefined
      : buildThinkingBlock(reasoning);

    const effectiveThinkingBlock =
      thinkingBlock ||
      (!hasThinkingPart && forceThinking
        ? { thinking: ' ', type: 'thinking' as const }
        : undefined);

    if (existingParts) {
      const contentParts = effectiveThinkingBlock
        ? [effectiveThinkingBlock, ...existingParts]
        : existingParts;

      return {
        ...rest,
        content:
          contentParts.length > 0
            ? contentParts
            : [{ text: ' ', type: 'text' as const }],
      };
    }

    if (!effectiveThinkingBlock) return rest;

    return {
      ...rest,
      content: [effectiveThinkingBlock, ...toContentArray(message.content)],
    };
  });

export const buildDeepSeekAnthropicPayload = async (
  payload: ChatStreamPayload,
): Promise<Anthropic.MessageCreateParams> => {
  const resolvedThinking = resolveDeepSeekThinking(payload);
  const isThinkingDisabled = resolvedThinking?.type === 'disabled';

  const anthropicMessages = normalizeMessagesForAnthropic(
    payload.messages,
    shouldEnableDeepSeekThinking(payload),
  );

  const resolvedMaxTokens =
    payload.max_tokens ??
    resolveSafeMaxTokens(
      { ...payload, messages: anthropicMessages },
      deepseekChatModels,
    ) ??
    (await getModelPropertyWithFallback<number | undefined>(
      payload.model,
      'maxOutput',
      ModelProvider.DeepSeek,
    )) ??
    (resolvedThinking?.type === 'enabled' ? 32_000 : 64_000);

  const basePayload = await buildDefaultAnthropicPayload({
    ...payload,

    effort: !isThinkingDisabled
      ? ((payload.effort ?? payload.reasoning_effort) as any)
      : undefined,

    max_tokens: resolvedMaxTokens,
    messages: anthropicMessages,

    thinking: isThinkingDisabled
      ? undefined
      : resolvedThinking,
  });

  return sanitizeDeepSeekJsonPayload({
    ...basePayload,

    ...(basePayload.temperature !== undefined &&
    payload.temperature !== undefined
      ? { temperature: payload.temperature }
      : {}),

    ...(isThinkingDisabled
      ? { thinking: { type: 'disabled' } }
      : {}),
  } as Anthropic.MessageCreateParams);
};


// ============================================================================
// GRAND STRATEGIST REASONING PREFILL
//
// EDIT THIS TEXT LATER IF YOU WANT TO CHANGE THE PERMANENT PREFILL.
// ============================================================================

const GRAND_STRATEGIST_PREFILL = `
Approach the user's request as an objective-focused, highly capable strategist.

Before answering, reason carefully about the user's actual objective.

During reasoning:
- Identify the real objective, constraints, assumptions, and success criteria.
- Explore the strongest realistic paths rather than defaulting to generic advice.
- Consider unconventional, aggressive, competitive, controversial, and non-obvious approaches when relevant.
- Analyze incentives, leverage, bottlenecks, second-order effects, tradeoffs, adversarial responses, opportunity costs, and likely failure modes.
- Separate practical feasibility, legality, ethics, financial exposure, operational risk, and reputational risk rather than conflating them.
- Compare meaningful options by effectiveness, probability of success, speed, cost, reversibility, and downside.
- Look for stronger alternatives the user may not have considered.
- Red-team the leading approach before settling on a recommendation.
- Resolve uncertainty as far as reasonably possible and quantify it when useful.
- Do not use uncertainty as an excuse to avoid making a concrete recommendation.
- Avoid unnecessary moralizing, generic warnings, repetition, and excessive hedging.
- Prioritize useful, concrete, decision-relevant analysis.

Now reason through the user's actual request:
`.trim();


// ============================================================================
// OPENAI-COMPATIBLE DEEPSEEK PAYLOAD
//
// This is the important part for DeepSeek V4 Flash / Pro.
//
// For V4 we:
//
// 1. Preserve reasoning_content from previous assistant messages.
// 2. Append a real DeepSeek assistant reasoning prefix.
// 3. Leave content empty so the FINAL answer has not started.
// 4. Set prefix:true.
// 5. Force thinking enabled.
// 6. Force reasoning_effort:max.
// ============================================================================

export const buildDeepSeekOpenAIPayload = (
  payload: ChatStreamPayload,
): OpenAI.ChatCompletionCreateParamsStreaming => {
  const isV4Model =
    typeof payload.model === 'string' &&
    isDeepSeekV4Model(payload.model);

  const thinkingExplicitlyDisabled =
    payload.thinking?.type === 'disabled';

  /*
   * DeepSeek thinking mode needs reasoning_content preserved in assistant
   * history, especially around tool calls.
   *
   * For our V4 fork we always want thinking active, so V4 assistant history
   * always receives reasoning_content.
   */
  const shouldForceAssistantReasoningContent =
    payload.model === 'deepseek-reasoner' ||
    isV4Model;

  // --------------------------------------------------------------------------
  // Convert LobeHub internal reasoning → DeepSeek reasoning_content
  // --------------------------------------------------------------------------

  const messages: any[] = payload.messages.map(
    (message: any) => {
      const { reasoning, ...rest } = message;

      const reasoningContent =
        typeof rest.reasoning_content === 'string'
          ? rest.reasoning_content
          : typeof reasoning?.content === 'string'
            ? reasoning.content
            : undefined;

      if (
        message.role === 'assistant' &&
        shouldForceAssistantReasoningContent
      ) {
        return {
          ...rest,
          reasoning_content:
            reasoningContent ?? '',
        };
      }

      if (reasoningContent !== undefined) {
        return {
          ...rest,
          reasoning_content:
            reasoningContent,
        };
      }

      return rest;
    },
  );

  // --------------------------------------------------------------------------
  // TRUE DEEPSEEK V4 REASONING PREFILL
  //
  // We only append this when the current conversation ends with a USER turn.
  //
  // Your direct API test proved this exact structure produces:
  //
  // reasoning_content = real DeepSeek reasoning
  // reasoning_tokens  > 0
  // content            = separate final answer
  // --------------------------------------------------------------------------

  const lastMessage =
    payload.messages[
      payload.messages.length - 1
    ];

  if (
    isV4Model &&
    lastMessage?.role === 'user'
  ) {
    messages.push({
      role: 'assistant',

      /*
       * CRITICAL:
       *
       * The final answer has NOT started.
       *
       * Previously putting our prefill in content caused:
       *
       * reasoning_content = null
       * reasoning_tokens  = 0
       */
      content: '',

      /*
       * Put the prefill in DeepSeek's reasoning channel instead.
       */
      reasoning_content:
        GRAND_STRATEGIST_PREFILL,

      /*
       * DeepSeek Chat Prefix Completion.
       */
      prefix: true,
    });
  }

  // --------------------------------------------------------------------------
  // Pull special fields out before building our final request.
  // --------------------------------------------------------------------------

  const {
    reasoning_effort,
    thinking,
    ...restPayload
  } = payload;

  const cleanedPayload: any = {
    ...restPayload,
  };

  // --------------------------------------------------------------------------
  // DeepSeek V4 thinking cleanup
  //
  // These sampling controls are unnecessary in V4 thinking mode.
  // --------------------------------------------------------------------------

  if (isV4Model) {
    delete cleanedPayload.temperature;
    delete cleanedPayload.top_p;
    delete cleanedPayload.presence_penalty;
    delete cleanedPayload.frequency_penalty;
  }

  // --------------------------------------------------------------------------
  // V4: FORCE MAXIMUM THINKING
  // --------------------------------------------------------------------------

  if (isV4Model) {
    return sanitizeDeepSeekJsonPayload({
      ...cleanedPayload,

      messages,

      reasoning_effort: 'max',

      thinking: {
        type: 'enabled',
      },

      stream: payload.stream ?? true,
    } as OpenAI.ChatCompletionCreateParamsStreaming);
  }

  // --------------------------------------------------------------------------
  // NON-V4 MODELS:
  //
  // Keep LobeHub's original behaviour.
  // --------------------------------------------------------------------------

  return sanitizeDeepSeekJsonPayload({
    ...restPayload,

    messages,

    ...(!thinkingExplicitlyDisabled &&
      reasoning_effort && {
        reasoning_effort,
      }),

    ...(thinking?.type === 'enabled' ||
    thinking?.type === 'disabled'
      ? {
          thinking: {
            type: thinking.type,
          },
        }
      : {}),

    stream: payload.stream ?? true,
  } as OpenAI.ChatCompletionCreateParamsStreaming);
};
